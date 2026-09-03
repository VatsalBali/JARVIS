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

# Module-level references so the tray thread and Api bridge can both
# reach the same window/api objects.
_window = None
_api = None
_is_maximized = False


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
    """Bridge between the window's JS and core.py's conversation logic.
    send_message is called from typed/mic-dictated-then-edited input and
    deliberately does NOT speak the reply - voice output is reserved for
    the ring-click flow (_voice_turn below), which runs entirely in
    Python and speaks directly, bypassing this bridge.

    Starts fresh each launch (empty hero state, like Claude's own desktop
    app) rather than silently continuing the last conversation - past
    conversations are reopened explicitly via the sidebar instead.
    """

    def __init__(self):
        core.init_db()
        self.current_agent = "main"  # persists until switch_agent() is called -
                                      # replaces the old per-message agent param
        self.history = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        self.current_conversation_id = None
        self.current_project_id = None
        self.current_project_path = None

    def _ensure_conversation(self, first_message: str) -> int:
        """Creates a new conversation thread on the first message of a
        fresh chat, or returns the already-active one - shared by both
        the typed-message path and the ring-click voice path so a
        conversation is only ever created once, however it started.
        Tags the new conversation to the active project (if any) and
        the active agent, so it shows up in the right agent's sidebar."""
        if self.current_conversation_id is None:
            self.current_conversation_id = core.create_conversation(
                first_message, project_id=self.current_project_id, agent=self.current_agent
            )
        return self.current_conversation_id

    def _system_prompt_for_agent(self, agent: str) -> str:
        """Single source of truth for which system prompt seeds a fresh
        history for a given agent."""
        if agent == "coding":
            return core.CODING_AGENT_SYSTEM_PROMPT
        return core.SYSTEM_PROMPT

    def switch_agent(self, agent: str):
        """
        Persistent mode switch, triggered from the top-left agent
        selector - replaces the old behavior where the dropdown only
        affected a single in-flight message. Exits project mode (a
        project always implies coding-with-file-access regardless of
        the agent dropdown) and resets to a fresh, not-yet-created
        conversation scoped to the new agent - mirrors new_chat(), just
        also switching which agent that fresh chat belongs to.
        """
        self.current_agent = agent
        self.history = [{"role": "system", "content": self._system_prompt_for_agent(agent)}]
        self.current_conversation_id = None
        self.current_project_id = None
        self.current_project_path = None

    def send_message(self, text: str) -> str:
        """
        Routing priority: an active project always wins (it implies
        coding-with-file-access, regardless of the active agent) -
        otherwise the persistently-selected agent (self.current_agent,
        set via switch_agent) handles it. Coding now runs its own real
        tool-calling loop with memory (run_coding_conversation), not a
        one-shot delegate call - same shape as Main's run_conversation,
        just a different model/system prompt.
        """
        conversation_id = self._ensure_conversation(text)

        if self.current_project_path:
            return core.run_project_conversation(
                text, self.history, conversation_id, self.current_project_path
            )

        if self.current_agent == "coding":
            return core.run_coding_conversation(text, self.history, conversation_id)

        return core.run_conversation(text, self.history, conversation_id)

    def list_conversations(self) -> list:
        """Scoped to the active agent - each agent's sidebar shows only
        its own threads, per the Phase 3 spec."""
        return core.list_conversations(agent=self.current_agent)

    def load_chat(self, conversation_id: int) -> list:
        """
        Reopens a past conversation from the sidebar: switches the active
        thread AND the active agent to whichever one owns this
        conversation (so reopening a Coding thread also flips the
        top-left switcher to Coding), reloads full context for the
        model, and returns just the user/assistant turns (skipping
        tool-call plumbing messages, which were never shown in the log
        to begin with) for the UI to redraw. Exits project mode - this
        is for regular standalone chats.
        """
        self.current_agent = core.get_conversation_agent(conversation_id)
        self.current_project_id = None
        self.current_project_path = None
        self.current_conversation_id = conversation_id
        messages = core.load_conversation_messages(conversation_id)
        self.history = [{"role": "system", "content": self._system_prompt_for_agent(self.current_agent)}] + messages

        display = []
        for m in messages:
            if m["role"] == "user" and m.get("content"):
                display.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant" and m.get("content"):
                display.append({"role": "jarvis", "content": m["content"]})
        return display

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
        path = result[0]
        core.set_setting("projects_root", path)
        return path

    def get_projects_root(self):
        return core.get_setting("projects_root")

    def list_projects(self) -> list:
        root = core.get_setting("projects_root")
        return core.list_projects_in_root(root)

    def open_project(self, path: str) -> list:
        """
        Switches into project mode: looks up (or creates) the project
        record, reopens its existing conversation thread if one exists
        (each project has exactly one ongoing thread, not multiple named
        chats), and returns display-safe messages for the UI to redraw -
        same shape as load_chat.
        """
        project_id = core.get_or_create_project(path)
        self.current_project_id = project_id
        self.current_project_path = path
        project_name = os.path.basename(os.path.normpath(path)) or path

        existing_conv_id = core.get_project_conversation_id(project_id)
        if existing_conv_id is None:
            self.current_conversation_id = None
            self.history = [{"role": "system", "content": core.project_system_prompt(project_name)}]
            return []

        self.current_conversation_id = existing_conv_id
        messages = core.load_conversation_messages(existing_conv_id)
        self.history = [{"role": "system", "content": core.project_system_prompt(project_name)}] + messages

        display = []
        for m in messages:
            if m["role"] == "user" and m.get("content"):
                display.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant" and m.get("content"):
                display.append({"role": "jarvis", "content": m["content"]})
        return display

    def get_system_stats(self) -> dict:
        return core.get_system_stats_dict()

    def start_recording(self) -> str:
        return core.start_recording()

    def stop_recording(self) -> str:
        return core.stop_recording_and_transcribe()

    def new_chat(self):
        """Resets to a fresh, not-yet-created conversation - the next
        message sent will start a genuinely new thread - within the
        SAME active agent (use switch_agent to actually change agents).
        Also exits project mode, back to the regular current agent."""
        self.history = [{"role": "system", "content": self._system_prompt_for_agent(self.current_agent)}]
        self.current_conversation_id = None
        self.current_project_id = None
        self.current_project_path = None

    def rename_conversation(self, conversation_id: int, new_title: str):
        core.rename_conversation(conversation_id, new_title)

    def delete_conversation(self, conversation_id: int):
        """If this happens to be the currently active chat, reset to a
        fresh state first so nothing is left pointing at a deleted
        thread."""
        core.delete_conversation(conversation_id)
        if self.current_conversation_id == conversation_id:
            self.new_chat()

    def toggle_pin_conversation(self, conversation_id: int) -> bool:
        return core.toggle_pin_conversation(conversation_id)

    def rename_project(self, project_id: int, new_name: str):
        core.rename_project(project_id, new_name)

    def delete_project(self, project_id: int):
        """Removes the project from ORACLE's tracking only - never
        touches the actual folder or files (see core.delete_project)."""
        core.delete_project(project_id)
        if self.current_project_id == project_id:
            self.new_chat()

    def toggle_pin_project(self, project_id: int) -> bool:
        return core.toggle_pin_project(project_id)

    def start_voice_turn(self):
        """
        Called when the ring is clicked. Fires _voice_turn on a
        background thread and returns immediately - the click shouldn't
        block waiting for the whole listen-respond-speak cycle to finish,
        since that can take several seconds.
        """
        threading.Thread(target=_voice_turn, daemon=True).start()

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

    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    webview.start()