"""
ORACLE - Background Desktop App

Runs fullscreen and borderless (no OS title bar), covering the whole
screen with the dashboard UI. Closing it (the "-" button in the top
right) just hides it to the system tray; ORACLE keeps running in the
background until you explicitly Quit from the tray menu.

Activation is wake-word based: say "Oracle" and it listens for whatever
you say next (see core.py's WAKE_WORD section for how this works and its
CPU trade-offs). There is no manual hotkey anymore - voice is the only
hands-free trigger, alongside the on-screen mic button and text input.

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
import time
import webview
import pystray
from PIL import Image, ImageDraw
import core

# Module-level references so the wake-word thread, tray thread, and Api
# bridge can all reach the same window/api objects.
_window = None
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

        # Clean up a leftover entry from before the JARVIS -> ORACLE rename,
        # if one exists from an earlier build.
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
    Bridge between index.html's JS and core.py's conversation logic - see
    core.py for the actual tool-calling / LLM logic. hide_window lets the
    dashboard's own "-" button hide the window to the tray, in addition
    to the tray menu's Hide option. There is no startup greeting anymore -
    ORACLE only greets you when the wake word activates it (see
    _wake_turn below).
    """

    def __init__(self):
        core.init_db()
        self.history = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        prior = core.load_recent_history()
        if prior:
            self.history.extend(prior)

    def send_message(self, text: str) -> str:
        reply = core.run_conversation(text, self.history)
        # Speak in the background rather than blocking this return - the
        # reply text should appear in chat immediately, with audio playing
        # alongside it, not only after playback finishes.
        threading.Thread(target=core.speak, args=(reply,), daemon=True).start()
        return reply

    def get_system_stats(self) -> dict:
        return core.get_system_stats_dict()

    def start_recording(self) -> str:
        return core.start_recording()

    def stop_recording(self) -> str:
        return core.stop_recording_and_transcribe()

    def hide_window(self):
        if _window:
            _window.hide()


def _run_js(script: str):
    """
    Runs JS in the already-loaded page - this is how Python pushes updates
    to the UI on its own initiative (window.evaluate_js), as opposed to
    the normal flow where JS calls into Python and waits for a return
    value. Needed here because wake-word activation happens from a
    background thread, not from a click inside the page.
    """
    if _window:
        try:
            _window.evaluate_js(script)
        except Exception as e:
            print(f"evaluate_js failed: {e}")


def _wake_turn():
    """
    The full wake-word-triggered interaction: show the window if it was
    hidden, greet the user (varied each time, via core.generate_wake_greeting),
    then immediately listen for their actual command, transcribe it, get
    ORACLE's reply, and push everything into the visible chat log.
    """
    if _window:
        _window.show()

    _run_js("document.getElementById('ringStatus').textContent = 'ACTIVATED';")

    greeting = core.generate_wake_greeting()
    _api.history.append({"role": "assistant", "content": greeting})
    _run_js(f"addMessage('jarvis', {json.dumps(greeting)});")
    core.speak(greeting)

    _run_js("document.getElementById('ringStatus').textContent = 'LISTENING...';")
    _run_js("document.getElementById('ring').classList.add('thinking');")

    text = core.listen_and_transcribe()

    if not text or text.startswith("Error"):
        print(f"Wake turn ended without a usable command: {text}")
        _run_js("document.getElementById('ring').classList.remove('thinking');")
        _run_js("document.getElementById('ringStatus').textContent = 'READY / AWAITING INPUT';")
        return

    _run_js(f"addMessage('user', {json.dumps(text)});")
    _run_js("document.getElementById('ringStatus').textContent = 'THINKING...';")

    reply = core.run_conversation(text, _api.history)

    _run_js(f"addMessage('jarvis', {json.dumps(reply)});")
    _run_js("document.getElementById('ringStatus').textContent = 'SPEAKING...';")
    core.speak(reply)

    _run_js("document.getElementById('ring').classList.remove('thinking');")
    _run_js("document.getElementById('ringStatus').textContent = 'READY / AWAITING INPUT';")


def run_wake_word_listener():
    """
    Runs on its own daemon thread. Starts the continuous background
    microphone stream, then periodically checks it for the wake word.
    While an actual voice turn is being handled, the wake-word stream is
    stopped (see _wake_turn's caller below) so the two don't fight over
    the microphone and so old buffered audio can't immediately re-trigger.
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


def on_closing():
    """
    Intercepts the window's close event. Returning False here cancels the
    actual close (this is a pywebview-specific convention - confirmed by
    reading pywebview's own source: a closing-event handler that returns
    False sets should_cancel=True internally). We hide instead, so ORACLE
    keeps running with just the tray icon left.
    """
    _window.hide()
    return False


def _make_tray_image():
    """
    Draws a simple glowing-ring icon in memory - no external .ico/.png
    file needed, keeping the project self-contained.
    """
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
    """
    Runs the system tray icon's event loop. This must run on its own
    thread since webview.start() below blocks the main thread for its
    own event loop - two GUI loops can't share one thread.
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

    _window = webview.create_window(
        "ORACLE",
        resource_path("index.html"),
        js_api=api,
        fullscreen=True,   # fills the entire screen
        frameless=True,    # no OS title bar/borders
        resizable=True,
    )

    _window.events.closing += on_closing

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    wake_thread = threading.Thread(target=run_wake_word_listener, daemon=True)
    wake_thread.start()

    webview.start()c