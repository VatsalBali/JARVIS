"""
ORACLE - Background Desktop App

Single window now (merged from the earlier HUD + separate chat window
split): a sidebar on the left, and a main area that shows a hero greeting
with the status ring and a big input box when there's no conversation
yet, collapsing into a normal scrolling chat view once you send a
message - similar in spirit to Claude's own desktop app. The ring stays
click-to-talk throughout, whether in the hero or the compact top-bar
version once a conversation is active.

The window hides to the tray rather than closing outright; ORACLE keeps
running in the background until you explicitly Quit from the tray menu.

Note: true window transparency was attempted but dropped - Windows'
WebView2 engine has ongoing, unresolved bugs rendering transparent
windows correctly (confirmed via multiple open upstream issues as of
2026), so this uses a solid opaque background instead.

SETUP:
    pip install -r requirements.txt
    python ui.py

Look for the ORACLE icon in your system tray (may be under the "^" hidden
icons arrow) to show/hide the window or quit for real.
"""

import threading
import os
import sys
import json
import webview
import pystray
from PIL import Image, ImageDraw
import core
import gemini_voice
from oracle_session import ChatSession

# Module-level references so the tray thread and Api bridge can both
# reach the same window/api objects.
_window = None
_api = None
_is_maximized = False

# Confirm-tier tool calls (core.TOOL_TIERS) waiting on the owner's Yes/No.
# request id -> {"event": threading.Event, "ok": bool}
_pending_confirms = {}
_pending_lock = threading.Lock()
CONFIRM_TIMEOUT_SEC = 120  # no answer counts as No


def resource_path(relative_path: str) -> str:
    """
    Resolves a bundled file's path correctly whether running as a plain
    script (python ui.py) or as a PyInstaller-frozen .exe. PyInstaller's
    --onefile mode extracts bundled data files (like index.html) to a
    temporary folder at runtime and exposes its path via sys._MEIPASS -
    without this, the packaged .exe would look for index.html next to
    itself and fail to find it.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def _ensure_autostart():
    """
    Self-registers ORACLE to launch automatically at Windows login, by
    writing to the current user's Run registry key (HKEY_CURRENT_USER -
    no admin rights required, unlike HKEY_LOCAL_MACHINE). Only does this
    when actually running as a packaged .exe (sys.frozen is set by
    PyInstaller) - registering the dev-mode `python ui.py` command would
    break the moment this project folder moves or Python is reinstalled,
    so autostart is deliberately tied to the built .exe, not the script.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        import winreg
        exe_path = sys.executable
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_READ,
        )

        try:
            winreg.DeleteValue(key, "JARVIS")
        except FileNotFoundError:
            pass

        try:
            current, _ = winreg.QueryValueEx(key, "ORACLE")
        except FileNotFoundError:
            current = None
        if current != exe_path:
            winreg.SetValueEx(key, "ORACLE", 0, winreg.REG_SZ, exe_path)
        winreg.CloseKey(key)
    except Exception as e:
        print(f"Could not register autostart: {e}")


class Api(ChatSession):
    """Bridge between the window's JS and core.py's conversation logic.
    Chat state and routing live in ChatSession (shared with the WebSocket
    backend, oracle_server.py); this adds only the window-specific methods.
    send_message deliberately does NOT speak the reply - voice output is
    reserved for the ring-click flow (_voice_turn below).
    """

    def choose_projects_root(self):
        """
        Opens the native OS folder picker so the user can point ORACLE at
        their projects folder (e.g. D:\\Projects). Persists the choice so
        it's remembered across restarts. Returns the chosen path, or None
        if the user cancelled.
        """
        if not _window:
            return None
        result = _window.create_file_dialog(webview.FileDialog.FOLDER)
        if not result:
            return None
        return self.set_projects_root(result[0])

    def start_voice_turn(self):
        """
        Called when the ring is clicked. Fires _voice_turn on a
        background thread and returns immediately - the click shouldn't
        block waiting for the whole listen-respond-speak cycle to finish,
        since that can take several seconds.
        """
        threading.Thread(target=_voice_turn, daemon=True).start()

    def confirm_reply(self, request_id: str, ok: bool):
        """Yes/No clicked on a confirmation card (see _ui_confirm)."""
        with _pending_lock:
            slot = _pending_confirms.get(request_id)
        if slot:
            slot["ok"] = bool(ok)
            slot["event"].set()

    def hide_window(self):
        """Hides the window to the tray - the "x" (close) button."""
        if _window:
            _window.hide()

    def minimize_window(self):
        if _window:
            _window.minimize()

    def toggle_maximize_window(self):
        """
        pywebview doesn't expose a reliable "is this window currently
        maximized" query, so we track it ourselves to know whether the
        next click should maximize or restore.
        """
        global _is_maximized
        if _window:
            if _is_maximized:
                _window.restore()
            else:
                _window.maximize()
            _is_maximized = not _is_maximized


def _run_js(script: str):
    """Pushes JS into the window from Python's own initiative - used
    during the ring-click voice flow, which happens on a background
    thread with no further click involved once started."""
    if _window:
        try:
            _window.evaluate_js(script)
        except Exception as e:
            print(f"evaluate_js failed: {e}")


def _ui_confirm(request: dict) -> bool:
    """
    core's confirmation handler. Called on whichever thread is running the
    tool call (a js_api thread for typed chat, the voice thread for ring
    clicks), so it can block: it shows a Yes/No card in the window and
    waits for Api.confirm_reply. Times out to No.
    """
    if not _window:
        return False
    slot = {"event": threading.Event(), "ok": False}
    with _pending_lock:
        _pending_confirms[request["id"]] = slot

    _window.show()
    _run_js(f"showConfirm({json.dumps(request)});")
    answered = slot["event"].wait(CONFIRM_TIMEOUT_SEC)

    with _pending_lock:
        _pending_confirms.pop(request["id"], None)
    if not answered:
        _run_js(f"expireConfirm({json.dumps(request['id'])});")
    return answered and slot["ok"]


def _voice_turn():
    """
    The full ring-click interaction: show the window if it was hidden,
    then run one streaming Gemini Live exchange - listening, thinking
    (including any tool calls), and speaking all happen inside that one
    call now, rather than as three separate local-STT / cloud-LLM /
    local-TTS steps. Both sides of the exchange still land in the
    visible chat log and the same SQLite conversation as before.
    """
    if _window:
        _window.show()

    _run_js("setRingStatus('LISTENING...'); setRingThinking(true);")

    def ensure_conversation(user_text: str) -> int:
        conv_id = _api._ensure_conversation(user_text)
        _run_js(f"addMessage('user', {json.dumps(user_text)});")
        return conv_id

    def on_status(status: str):
        _run_js(f"setRingStatus('{status}');")

    user_text, reply_text = gemini_voice.voice_turn_live(
        _api.history, ensure_conversation, on_status=on_status
    )

    if not user_text:
        print("Voice turn ended without a usable command.")
        _run_js("setRingThinking(false); setRingStatus('');")
        return

    _run_js("if (typeof refreshChatList === 'function') refreshChatList();")
    if reply_text:
        _run_js(f"addMessage('jarvis', {json.dumps(reply_text)});")

    _run_js("setRingThinking(false); setRingStatus('');")


def on_closing():
    """
    Intercepts the window's close event. Returning False cancels the
    actual close (confirmed via pywebview's own source: a closing-event
    handler that returns False sets should_cancel=True internally) - we
    hide instead, so ORACLE keeps running with just the tray icon left.
    """
    _window.hide()
    return False


def _make_tray_image():
    """Draws a simple glowing-ring icon in memory - no external .ico/.png
    file needed, keeping the project self-contained."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((6, 6, 58, 58), outline=(95, 216, 232, 255), width=5)
    draw.ellipse((24, 24, 40, 40), fill=(95, 216, 232, 255))
    return img


def _tray_show(icon, item):
    if _window:
        _window.show()


def _tray_hide(icon, item):
    if _window:
        _window.hide()


def _tray_quit(icon, item):
    icon.stop()
    if _window:
        _window.destroy()


def run_tray():
    """Runs the system tray icon's event loop. This must run on its own
    thread since webview.start() below blocks the main thread for its
    own event loop - two GUI loops can't share one thread."""
    icon = pystray.Icon(
        "oracle",
        _make_tray_image(),
        "ORACLE",
        menu=pystray.Menu(
            pystray.MenuItem("Show", _tray_show, default=True),
            pystray.MenuItem("Hide", _tray_hide),
            pystray.MenuItem("Quit", _tray_quit),
        ),
    )
    icon.run()


if __name__ == "__main__":
    _ensure_autostart()

    api = Api()
    _api = api

    _window = webview.create_window(
        "ORACLE",
        resource_path("index.html"),
        js_api=api,
        width=1200,
        height=780,
        min_size=(760, 520),
        frameless=True,     # no OS title bar/borders - custom "-" button instead
        easy_drag=False,    # whole-window drag was the bug - drag now comes from
                             # the .pywebview-drag-region class in index.html instead,
                             # scoped to just the topbar/sidebar-top strips
        text_select=True,   # was blocking text selection/copy app-wide
        resizable=True,
        shadow=True,        # subtle drop shadow - Windows only, floating-HUD feel
    )
    _window.events.closing += on_closing
    core.set_confirm_handler(_ui_confirm)

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    webview.start()
