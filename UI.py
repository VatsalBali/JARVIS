"""
JARVIS - Background Desktop App

Runs fullscreen and borderless (no OS title bar), covering the whole
screen with the dashboard UI. Closing it (the "-" button in the top
right) just hides it to the system tray; JARVIS keeps running in the
background until you explicitly Quit from the tray menu.

Note: true window transparency was attempted but dropped - Windows'
WebView2 engine has ongoing, unresolved bugs rendering transparent
windows correctly (confirmed via multiple open upstream issues as of
2026), so this uses a solid opaque background instead.

SETUP:
    pip install -r requirements.txt
    python ui.py

Look for the JARVIS icon in your system tray (may be under the "^" hidden
icons arrow) to show/hide the window or quit for real.
"""

import threading
import os
import sys
import webview
import pystray
from PIL import Image, ImageDraw
import core

# Module-level reference so the tray thread and the Api bridge can both
# reach the same window object to show/hide/destroy it.
_window = None


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
    Self-registers JARVIS to launch automatically at Windows login, by
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
            current, _ = winreg.QueryValueEx(key, "JARVIS")
        except FileNotFoundError:
            current = None
        if current != exe_path:
            winreg.SetValueEx(key, "JARVIS", 0, winreg.REG_SZ, exe_path)
        winreg.CloseKey(key)
    except Exception as e:
        print(f"Could not register autostart: {e}")


class Api:
    """
    Bridge between index.html's JS and core.py's conversation logic - see
    core.py for the actual tool-calling / LLM logic. hide_window lets the
    dashboard's own "-" button hide the window to the tray, in addition
    to the tray menu's Hide option.
    """

    def __init__(self):
        core.init_db()
        self.history = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        prior = core.load_recent_history()
        if prior:
            self.history.extend(prior)
        self.resuming = bool(prior)

    def get_greeting(self) -> str:
        greeting = core.generate_greeting(resuming=self.resuming)
        self.history.append({"role": "assistant", "content": greeting})
        return greeting

    def send_message(self, text: str) -> str:
        return core.run_conversation(text, self.history)

    def get_system_stats(self) -> dict:
        return core.get_system_stats_dict()

    def hide_window(self):
        if _window:
            _window.hide()


def on_closing():
    """
    Intercepts the window's close event. Returning False here cancels the
    actual close (this is a pywebview-specific convention - confirmed by
    reading pywebview's own source: a closing-event handler that returns
    False sets should_cancel=True internally). We hide instead, so JARVIS
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
        "jarvis",
        _make_tray_image(),
        "J.A.R.V.I.S.",
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

    _window = webview.create_window(
        "J.A.R.V.I.S.",
        resource_path("index.html"),
        js_api=api,
        fullscreen=True,   # fills the entire screen
        frameless=True,    # no OS title bar/borders
        resizable=True,
    )

    _window.events.closing += on_closing

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    webview.start()