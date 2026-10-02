"""
ORACLE - chat session state, independent of any window toolkit.

Holds the active agent, conversation and project, and routes messages to
the right core.run_*_conversation loop. Used by UI.py's pywebview Api
(which adds the window-only methods on top) and by oracle_server.py's
WebSocket RPC, so the chat window behaves the same over either bridge.
"""

import os
import core


class ChatSession:
    """Starts fresh each launch (empty hero state) rather than silently
    continuing the last conversation - past conversations are reopened
    explicitly via the sidebar instead."""

    def __init__(self):
        core.init_db()
        self.current_agent = "main"
        self.history = [{"role": "system", "content": core.SYSTEM_PROMPT}]
        self.current_conversation_id = None
        self.current_project_id = None
        self.current_project_path = None

    def _ensure_conversation(self, first_message: str) -> int:
        """Creates a new conversation thread on the first message of a
        fresh chat, or returns the already-active one - shared by the typed
        path and the voice path so a conversation is only ever created once,
        however it started. Tagged to the active project and agent."""
        if self.current_conversation_id is None:
            self.current_conversation_id = core.create_conversation(
                first_message, project_id=self.current_project_id, agent=self.current_agent
            )
        return self.current_conversation_id

    def _system_prompt_for_agent(self, agent: str) -> str:
        if agent == "coding":
            return core.CODING_AGENT_SYSTEM_PROMPT
        return core.SYSTEM_PROMPT

    @staticmethod
    def _display_messages(messages: list) -> list:
        """User/assistant turns only, skipping tool-call plumbing."""
        display = []
        for m in messages:
            if m["role"] == "user" and m.get("content"):
                display.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant" and m.get("content"):
                display.append({"role": "jarvis", "content": m["content"]})
        return display

    def switch_agent(self, agent: str):
        """Persistent mode switch from the agent selector. Exits project
        mode and resets to a fresh conversation scoped to the new agent."""
        self.current_agent = agent
        self.history = [{"role": "system", "content": self._system_prompt_for_agent(agent)}]
        self.current_conversation_id = None
        self.current_project_id = None
        self.current_project_path = None

    def send_message(self, text: str) -> str:
        """An active project always wins (coding with file access);
        otherwise the selected agent handles it."""
        conversation_id = self._ensure_conversation(text)

        if self.current_project_path:
            return core.run_project_conversation(
                text, self.history, conversation_id, self.current_project_path
            )
        if self.current_agent == "coding":
            return core.run_coding_conversation(text, self.history, conversation_id)
        return core.run_conversation(text, self.history, conversation_id, context=core.screen.context_line())

    def list_conversations(self) -> list:
        return core.list_conversations(agent=self.current_agent)

    def load_chat(self, conversation_id: int) -> list:
        """Reopens a past conversation: switches the active thread and agent
        to the one that owns it and reloads full context for the model."""
        self.current_agent = core.get_conversation_agent(conversation_id)
        self.current_project_id = None
        self.current_project_path = None
        self.current_conversation_id = conversation_id
        messages = core.load_conversation_messages(conversation_id)
        self.history = [{"role": "system", "content": self._system_prompt_for_agent(self.current_agent)}] + messages
        return self._display_messages(messages)

    def set_projects_root(self, path: str) -> str:
        core.set_setting("projects_root", path)
        return path

    def get_projects_root(self):
        return core.get_setting("projects_root")

    def list_projects(self) -> list:
        return core.list_projects_in_root(core.get_setting("projects_root"))

    def open_project(self, path: str) -> list:
        """Switches into project mode, reopening the project's single
        ongoing thread if it has one."""
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
        return self._display_messages(messages)

    def get_system_stats(self) -> dict:
        return core.get_system_stats_dict()

    def start_recording(self) -> str:
        return core.start_recording()

    def stop_recording(self) -> str:
        return core.stop_recording_and_transcribe()

    def new_chat(self):
        """Fresh, not-yet-created conversation within the same agent; also
        exits project mode."""
        self.history = [{"role": "system", "content": self._system_prompt_for_agent(self.current_agent)}]
        self.current_conversation_id = None
        self.current_project_id = None
        self.current_project_path = None

    def rename_conversation(self, conversation_id: int, new_title: str):
        core.rename_conversation(conversation_id, new_title)

    def delete_conversation(self, conversation_id: int):
        core.delete_conversation(conversation_id)
        if self.current_conversation_id == conversation_id:
            self.new_chat()

    def toggle_pin_conversation(self, conversation_id: int) -> bool:
        return core.toggle_pin_conversation(conversation_id)

    def rename_project(self, project_id: int, new_name: str):
        core.rename_project(project_id, new_name)

    def delete_project(self, project_id: int):
        """Removes the project from ORACLE's tracking only - never touches
        the actual folder or files."""
        core.delete_project(project_id)
        if self.current_project_id == project_id:
            self.new_chat()

    def toggle_pin_project(self, project_id: int) -> bool:
        return core.toggle_pin_project(project_id)
