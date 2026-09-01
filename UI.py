"""
ORACLE - Background Desktop App

Two separate windows now:
  - The HUD (index.html): fullscreen, borderless, shows the status ring.
    Activated by the wake word ("Oracle") - listens, replies, and SPEAKS
    the reply out loud.
  - The Chat window (chat.html): a normal-sized window you open via the
    chat icon in the HUD's top bar. Typed (or mic-dictated-then-edited)
    conversations here stay TEXT-ONLY - ORACLE never speaks a reply to
    something you typed, since you're already looking at the screen.

Both windows hide to the tray rather than closing outright; ORACLE keeps
running in the background until you explicitly Quit from the tray menu.

Note: true window transparency was attempted but dropped - Windows'
WebView2 engine has ongoing, unresolved bugs rendering transparent
windows correctly (confirmed via multiple open upstream issues as of
2026), so both windows use a solid opaque background instead.

SETUP:
    pip install -r requirements.txt
    python ui.py

Look for the ORACLE icon in your system tray (may be under the "^" hidden
icons arrow) to show/hide the HUD or quit for real.
"""

import threading
import os
import sys
import json
import time
import webview
import pystray
from PIL import Image, ImageDraw
import core

# Module-level references so the wake-word thread, tray thread, and Api
# bridge can all reach the same window/api objects.
_hud_window = None
_chat_window = None
_api = None


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


class Api:
    """
    Bridge between both windows' JS and core.py's conversation logic.
    send_message is called only from the chat window and deliberately
    does NOT speak the reply - voice output is reserved for the wake-word
    flow (_wake_turn below), which runs entirely in Python and speaks
    directly, bypassing this bridge.
    """

    def __init__(self):
        core.init_db()
        self.history = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        prior = core.load_recent_history()
        if prior:
            self.history.extend(prior)

    def send_message(self, text: str) -> str:
        return core.run_conversation(text, self.history)

    def get_system_stats(self) -> dict:
        return core.get_system_stats_dict()

    def start_recording(self) -> str:
        return core.start_recording()

    def stop_recording(self) -> str:
        return core.stop_recording_and_transcribe()

    def hide_window(self):
        """Hides the HUD window - called by the HUD's own "-" button."""
        if _hud_window:
            _hud_window.hide()

    def show_chat_window(self):
        """Shows the chat window - called by the HUD's chat-icon button."""
        if _chat_window:
            _chat_window.show()

    def hide_chat_window(self):
        """Hides the chat window - called by the chat window's own "-" button."""
        if _chat_window:
            _chat_window.hide()


def _run_js_hud(script: str):
    """Pushes JS into the HUD window from Python's own initiative - used
    during wake-word activation, which happens on a background thread
    with no click involved."""
    if _hud_window:
        try:
            _hud_window.evaluate_js(script)
        except Exception as e:
            print(f"evaluate_js (HUD) failed: {e}")


def _run_js_chat(script: str):
    """Same as _run_js_hud, but for the chat window - used to log the
    wake-word conversation into the chat transcript even when that window
    isn't currently visible, so the history is there if you open it later."""
    if _chat_window:
        try:
            _chat_window.evaluate_js(script)
        except Exception as e:
            print(f"evaluate_js (chat) failed: {e}")


def _wake_turn():
    """
    The full wake-word-triggered interaction: show the HUD if it was
    hidden, greet the user (varied each time), listen for their command,
    transcribe it, get ORACLE's reply - speaking both the greeting and
    the reply out loud - while also logging the exchange into the chat
    window's transcript (without forcing that window to open).
    """
    if _hud_window:
        _hud_window.show()

    _run_js_hud("document.getElementById('ringStatus').textContent = 'ACTIVATED';")

    greeting = core.generate_wake_greeting()
    _api.history.append({"role": "assistant", "content": greeting})
    _run_js_chat(f"addMessage('jarvis', {json.dumps(greeting)});")
    core.speak(greeting)

    _run_js_hud("document.getElementById('ringStatus').textContent = 'LISTENING...';")
    _run_js_hud("document.getElementById('ring').classList.add('thinking');")

    text = core.listen_and_transcribe()

    if not text or text.startswith("Error"):
        print(f"Wake turn ended without a usable command: {text}")
        _run_js_hud("document.getElementById('ring').classList.remove('thinking');")
        _run_js_hud("document.getElementById('ringStatus').textContent = 'READY / AWAITING INPUT';")
        return

    _run_js_chat(f"addMessage('user', {json.dumps(text)});")
    _run_js_hud("document.getElementById('ringStatus').textContent = 'THINKING...';")

    reply = core.run_conversation(text, _api.history)

    _run_js_chat(f"addMessage('jarvis', {json.dumps(reply)});")
    _run_js_hud("document.getElementById('ringStatus').textContent = 'SPEAKING...';")
    core.speak(reply)

    _run_js_hud("document.getElementById('ring').classList.remove('thinking');")
    _run_js_hud("document.getElementById('ringStatus').textContent = 'READY / AWAITING INPUT';")


def run_wake_word_listener():
    """
    Runs on its own daemon thread. Starts the continuous background
    microphone stream, then periodically checks it for the wake word.
    While an actual voice turn is being handled, the wake-word stream is
    stopped so the two don't fight over the microphone and so old
    buffered audio can't immediately re-trigger.
    """
    core.start_wake_word_stream()
    try:
        while True:
            time.sleep(core.WAKE_CHECK_INTERVAL_SEC)
            if core.check_for_wake_word():
                core.stop_wake_word_stream()
                try:
                    _wake_turn()
                finally:
                    core.start_wake_word_stream()
    except Exception as e:
        print(f"Wake word listener crashed: {e}")


def on_hud_closing():
    """
    Intercepts the HUD window's close event. Returning False cancels the
    actual close (confirmed via pywebview's own source: a closing-event
    handler that returns False sets should_cancel=True internally) - we
    hide instead, so ORACLE keeps running with just the tray icon left.
    """
    _hud_window.hide()
    return False


def on_chat_closing():
    """Same idea as on_hud_closing, for the chat window."""
    _chat_window.hide()
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
    if _hud_window:
        _hud_window.show()


def _tray_hide(icon, item):
    if _hud_window:
        _hud_window.hide()


def _tray_quit(icon, item):
    icon.stop()
    if _chat_window:
        _chat_window.destroy()
    if _hud_window:
        _hud_window.destroy()


def run_tray():
    """
    Runs the system tray icon's event loop. This must run on its own
    thread since webview.start() below blocks the main thread for its
    own event loop - two GUI loops can't share one thread. Tray Show/Hide
    controls the HUD only; the chat window has its own open/hide controls
    (the HUD's chat icon, and the chat window's own "-" button).
    """
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

    _hud_window = webview.create_window(
        "ORACLE",
        resource_path("index.html"),
        js_api=api,
        fullscreen=True,   # fills the entire screen
        frameless=True,    # no OS title bar/borders
        resizable=True,
    )
    _hud_window.events.closing += on_hud_closing

    _chat_window = webview.create_window(
        "ORACLE - Chat",
        resource_path("chat.html"),
        js_api=api,
        width=420,
        height=640,
        min_size=(320, 400),
        frameless=True,
        easy_drag=True,   # not fullscreen, so unlike the HUD this one benefits
                           # from being draggable to reposition it
        resizable=True,
        hidden=True,       # starts closed - opened via the HUD's chat icon
    )
    _chat_window.events.closing += on_chat_closing

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    wake_thread = threading.Thread(target=run_wake_word_listener, daemon=True)
    wake_thread.start()

    webview.start()