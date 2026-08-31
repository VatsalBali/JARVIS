"""
JARVIS - Desktop UI

Renders index.html (a real HTML/CSS/JS file) inside a native window using
pywebview - no browser tabs/address bar, just your JARVIS interface. The
`Api` class below is the bridge: JavaScript in index.html calls its methods
directly (via window.pywebview.api.methodName(...)), and those methods call
straight into the same core.py functions the terminal version used.

SETUP:
    pip install pywebview
    python ui.py
"""

import webview
import core


class Api:
    """
    Every public method here becomes callable from JavaScript as
    window.pywebview.api.<method_name>(...). This is the ONLY communication
    channel between the HTML/JS frontend and the Python backend - the UI
    never touches the Groq API, the database, or any tool directly. It just
    calls these two methods and displays whatever comes back.
    """

    def __init__(self):
        core.init_db()
        self.history = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        prior = core.load_recent_history()
        if prior:
            self.history.extend(prior)
        self.resuming = bool(prior)

    def get_greeting(self) -> str:
        """Called once when the page finishes loading."""
        greeting = core.generate_greeting(resuming=self.resuming)
        self.history.append({"role": "assistant", "content": greeting})
        return greeting

    def send_message(self, text: str) -> str:
        """Called every time the user submits a message in the UI."""
        return core.run_conversation(text, self.history)

    def get_system_stats(self) -> dict:
        """
        Called directly by the sidebar's polling loop - deliberately NOT
        routed through the LLM/tool-calling path. Refreshing a sidebar bar
        every couple of seconds has nothing to do with conversation, so
        this calls core's stats function straight, with zero API calls
        and zero token cost.
        """
        return core.get_system_stats_dict()


if __name__ == "__main__":
    api = Api()
    window = webview.create_window(
        "J.A.R.V.I.S.",
        "index.html",
        js_api=api,
        width=900,
        height=700,
        min_size=(500, 400),
        background_color="#0a0e14",
    )
    webview.start()