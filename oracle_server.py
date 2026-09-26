"""
ORACLE - backend process with a WebSocket bridge (README section 4.3).

Runs core.py, gemini_voice.py and (later) the wake-word loop in one Python
process, and talks to the Electron orb and chat windows over a WebSocket
bound to 127.0.0.1 only. Every launch gets a random token; clients must
connect with ws://127.0.0.1:<port>/?token=<token> or are refused before the
upgrade. On startup one line is printed to stdout for the parent (Electron)
to read:

    ORACLE_READY {"port": 51234, "token": "..."}

Events to the UI:
    {"type":"hello","state":"asleep","muted":false,"wake_phrase":"Hey Jarvis","voice_enrolled":false}
    {"type":"state","state":"asleep|wake|listening|thinking|speaking|followup","label":"optional override"}
    {"type":"speaker","verified":true,"enrolled":true,"score":0.83}
    {"type":"enroll","ok":true}
    {"type":"level","value":0.0-1.0}
    {"type":"transcript","text":"...","final":false}
    {"type":"caption","text":"..."}
    {"type":"tool","name":"list_upcoming_events","label":"Checking your calendar…"}
    {"type":"confirm","id":"...","action":"...","details":"...","tier":"confirm|warn","spoken":true}
    {"type":"confirm_closed","id":"...","ok":false,"reason":"timeout|cancel|answered|spoken"}
    {"type":"turn","user":"...","reply":"..."}          a finished voice turn
    {"type":"muted","value":true}
    {"type":"open_chat"}
    {"type":"result","id":"...","ok":true,"value":...}  reply to a "call"

Events from the UI:
    {"type":"confirm_reply","id":"...","ok":true}
    {"type":"cancel"}
    {"type":"open_chat"}
    {"type":"mute","value":true}
    {"type":"talk"}                                      hotkey / tray (same as the wake word)
    {"type":"enroll"}                                    tray "Learn my voice"
    {"type":"call","id":"...","method":"send_message","args":{"text":"hi"}}

Run standalone for testing:  python oracle_server.py [--port 8770]
"""

import argparse
import asyncio
import json
import os
import secrets
import sys
import threading
import time
from datetime import datetime, timedelta
from http import HTTPStatus
from urllib.parse import urlparse, parse_qs

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import core
from oracle_session import ChatSession

HOST = "127.0.0.1"
CONFIRM_TIMEOUT_SEC = 120  # no answer counts as No

# The chat window's former pywebview Api surface, now callable over the
# socket. Anything not listed here is refused.
RPC_METHODS = {
    "send_message", "switch_agent", "new_chat",
    "list_conversations", "load_chat", "rename_conversation",
    "delete_conversation", "toggle_pin_conversation",
    "get_projects_root", "set_projects_root", "list_projects", "open_project",
    "rename_project", "delete_project", "toggle_pin_project",
    "get_system_stats", "start_recording", "stop_recording",
}


MISSED_REMINDER_HOURS = 12  # older ones missed while ORACLE was off are dropped


def _seconds_since_input() -> float:
    """Seconds since the last keyboard/mouse input anywhere in Windows."""
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO(cbSize=ctypes.sizeof(LASTINPUTINFO))
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    return ((ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def _reminder_speech(kind: str, message: str, due: datetime, late: timedelta) -> str:
    if late > timedelta(minutes=2):
        return f"While you were away, Sir: at {due:%H:%M} you asked me to remind you: {message}."
    if kind == "timer":
        return f"Time's up, Sir. Your {message} is done."
    return f"Sir, a reminder: {message}."


class Backend:
    def __init__(self, port: int = 0):
        self.port = port
        self.token = secrets.token_urlsafe(24)
        self.session = ChatSession()
        self.clients = set()
        self.loop = None
        self.state = "asleep"
        self.muted = False
        self._pending = {}  # confirm id -> {"event": threading.Event, "ok": bool, "reason": str}
        self._pending_lock = threading.Lock()
        self._tasks = set()  # in-flight RPC tasks, kept so they aren't garbage-collected
        self.voice = None    # VoiceEngine, started by start_voice()
        self._enroll_after_conversation = False

        core.set_confirm_handler(self.confirm)
        core.set_tool_listener(lambda name, label: self.emit({"type": "tool", "name": name, "label": label}))

    # ---- events out (safe to call from any thread) ----

    def emit(self, event: dict):
        if event.get("type") == "state":
            self.state = event["state"]
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict):
        message = json.dumps(event)
        for ws in list(self.clients):
            try:
                await ws.send(message)
            except ConnectionClosed:
                pass

    # ---- confirmations (core's confirm handler; runs on a worker thread) ----

    def confirm(self, request: dict) -> bool:
        if not self.clients:
            return False  # nobody to ask, so the answer is no
        slot = {"event": threading.Event(), "ok": False, "reason": "timeout"}
        with self._pending_lock:
            self._pending[request["id"]] = slot
        spoken = bool(self.voice and self.voice.spoken_confirmation_allowed(request))
        self.emit({
            "type": "confirm",
            "id": request["id"],
            "action": request["title"],
            "details": request["details"],
            "tier": request["tier"],
            "spoken": spoken,  # the orb hints "say yes or no"
        })
        stop_listening = threading.Event()
        if spoken:
            # Whichever comes first, a click or a spoken yes/no, answers it.
            threading.Thread(
                target=self.voice.listen_yes_no,
                args=(stop_listening, lambda ok: self._answer_confirm(request["id"], ok, "spoken")),
                daemon=True,
            ).start()
        slot["event"].wait(CONFIRM_TIMEOUT_SEC)
        stop_listening.set()
        with self._pending_lock:
            self._pending.pop(request["id"], None)
        self.emit({"type": "confirm_closed", "id": request["id"], "ok": slot["ok"], "reason": slot["reason"]})
        return slot["ok"]

    def _answer_confirm(self, confirm_id: str, ok: bool, reason: str = "answered"):
        with self._pending_lock:
            slot = self._pending.get(confirm_id)
        if slot and not slot["event"].is_set():
            slot["ok"], slot["reason"] = bool(ok), reason
            slot["event"].set()

    def _cancel_all_confirms(self):
        with self._pending_lock:
            ids = list(self._pending)
        for confirm_id in ids:
            self._answer_confirm(confirm_id, False, "cancel")

    # ---- voice ----

    def _register_voice_tools(self):
        """"Oracle, learn my voice" (README 4.3). Confirm tier, and a warning
        when it would replace an existing voiceprint, so nobody else can
        quietly enrol their own voice."""
        def learn_owner_voice() -> str:
            if self.voice is None:
                return "Voice features aren't running."
            self._enroll_after_conversation = True
            self.voice.cancel()
            return "Enrolment will start as soon as this conversation closes. Tell the owner to read the passage on screen."

        core.AVAILABLE_FUNCTIONS["learn_owner_voice"] = learn_owner_voice
        if not any(t["function"]["name"] == "learn_owner_voice" for t in core.TOOLS):
            core.TOOLS.append({
                "type": "function",
                "function": {
                    "name": "learn_owner_voice",
                    "description": (
                        "Record about half a minute of the owner's speech to learn their voice, "
                        "so ORACLE can tell them apart from other people. Use when the owner says "
                        "something like 'learn my voice'."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        core.TOOL_TIERS["learn_owner_voice"] = {
            "tier": core.TIER_CONFIRM,
            "describe": lambda _a: (
                ("Replace the owner's saved voiceprint?", "Anyone who enrols here becomes the voice ORACLE obeys.")
                if self.voice and self.voice.voice_id.enrolled
                else ("Learn your voice now?", "You'll read a short passage aloud for about half a minute.")
            ),
        }
        core.TOOL_LABELS["learn_owner_voice"] = "Getting ready to learn your voice…"

    def _on_voice_state(self, event: dict):
        """Runs deferred enrolment once a conversation has gone to sleep."""
        if event.get("type") == "state" and event.get("state") == "asleep" and self._enroll_after_conversation:
            self._enroll_after_conversation = False
            threading.Timer(0.5, self.voice.start_enrollment).start()

    def _emit_voice(self, event: dict):
        self.emit(event)
        self._on_voice_state(event)

    # ---- timers & reminders ----

    def start_reminders(self):
        threading.Thread(target=self._reminder_loop, daemon=True, name="reminders").start()

    def _reminder_loop(self):
        """Fires due timers/reminders: a toast right away, plus a chime and
        speech through the orb as soon as nothing else is using it (never
        over a conversation). Ones missed while ORACLE was off are announced
        at startup if they're less than MISSED_REMINDER_HOURS old."""
        announced_toast = set()
        while True:
            try:
                for rid, kind, message, due in core.due_reminders():
                    late = datetime.now() - due
                    if late > timedelta(hours=MISSED_REMINDER_HOURS):
                        core.mark_reminder(rid, "missed")
                        continue
                    title = "Timer" if kind == "timer" else "Reminder"
                    if rid not in announced_toast:
                        core.send_notification(f"ORACLE {title.lower()}", message)
                        announced_toast.add(rid)
                    text = _reminder_speech(kind, message, due, late)
                    if self.voice is None or not self.voice.mic.running:
                        core.mark_reminder(rid, "fired")  # no voice: the toast is it
                        self.emit({"type": "caption", "text": text})
                    elif self.voice.announce(text, label=title.upper()):
                        core.mark_reminder(rid, "fired")
                    break  # one at a time; the next waits for this announcement
            except Exception as e:
                print(f"Reminder check failed: {e}", file=sys.stderr)
            time.sleep(1)

    # ---- morning briefing ----

    def start_briefing(self):
        threading.Thread(target=self._briefing_loop, daemon=True, name="briefing").start()

    def _briefing_loop(self):
        """Delivers the morning briefing once a day, the first time the owner
        is at the PC (keyboard/mouse input in the last minute) between
        briefing_after and briefing_until. Settings: briefing_auto on/off,
        briefing_after '07:30', briefing_until '12:00'."""
        while True:
            try:
                if self._briefing_due():
                    facts = core.get_briefing()
                    if self.voice.announce(facts, label="GOOD MORNING", compose=True):
                        core.set_setting("briefing_last_date", datetime.now().date().isoformat())
            except Exception as e:
                print(f"Briefing check failed: {e}", file=sys.stderr)
            time.sleep(20)

    def _briefing_due(self) -> bool:
        if (core.get_setting("briefing_auto") or "on").lower() != "on":
            return False
        if self.voice is None or self.voice.muted or self.voice.conversation is not None:
            return False
        now = datetime.now()
        if core.get_setting("briefing_last_date") == now.date().isoformat():
            return False
        after = datetime.strptime(core.get_setting("briefing_after") or "07:30", "%H:%M").time()
        until = datetime.strptime(core.get_setting("briefing_until") or "12:00", "%H:%M").time()
        return after <= now.time() < until and _seconds_since_input() < 60

    def start_voice(self):
        from voice_engine import VoiceEngine  # heavy imports: only when voice is used
        self.voice = VoiceEngine(self._emit_voice, self.session)
        self._register_voice_tools()
        self.voice.start()

    # ---- socket ----

    def _check_token(self, connection, request):
        token = parse_qs(urlparse(request.path).query).get("token", [""])[0]
        if not secrets.compare_digest(token, self.token):
            return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
        return None

    async def _handle_call(self, ws, msg: dict):
        call_id = msg.get("id")
        method = msg.get("method")
        args = msg.get("args") or {}
        if method not in RPC_METHODS or not isinstance(args, dict):
            reply = {"type": "result", "id": call_id, "ok": False, "error": f"Unknown method '{method}'."}
        else:
            try:
                # Worker thread: send_message can block for a long time
                # (model calls, and confirmations waiting on the owner).
                value = await asyncio.to_thread(getattr(self.session, method), **args)
                reply = {"type": "result", "id": call_id, "ok": True, "value": value}
            except Exception as e:
                reply = {"type": "result", "id": call_id, "ok": False, "error": str(e)}
        try:
            await ws.send(json.dumps(reply, default=str))
        except ConnectionClosed:
            pass

    async def _handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps({
                "type": "hello",
                "state": self.state,
                "muted": self.muted,
                "wake_phrase": self.voice.wake_phrase if self.voice else None,
                "voice_enrolled": bool(self.voice and self.voice.voice_id.enrolled),
            }))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")
                if kind == "confirm_reply":
                    self._answer_confirm(str(msg.get("id")), bool(msg.get("ok")))
                elif kind == "cancel":
                    self._cancel_all_confirms()
                    if self.voice:
                        self.voice.cancel()
                elif kind == "open_chat":
                    self.emit({"type": "open_chat"})
                elif kind == "mute":
                    self.muted = bool(msg.get("value"))
                    if self.voice:
                        await asyncio.to_thread(self.voice.set_muted, self.muted)
                    self.emit({"type": "muted", "value": self.muted})
                elif kind == "talk":
                    if self.voice:
                        self.voice.trigger()
                elif kind == "enroll":
                    # From the tray: a click is the owner's authority.
                    if self.voice:
                        self.voice.start_enrollment()
                elif kind == "call":
                    # Own task so a long send_message doesn't block
                    # confirm_reply messages arriving on this socket.
                    task = asyncio.create_task(self._handle_call(ws, msg))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        async with serve(self._handler, HOST, self.port, process_request=self._check_token) as server:
            port = server.sockets[0].getsockname()[1]
            print("ORACLE_READY " + json.dumps({"port": port, "token": self.token}), flush=True)
            await asyncio.Future()  # run until the process is stopped


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="ORACLE backend")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--no-voice", action="store_true", help="don't open the microphone (testing)")
    parser.add_argument(
        "--parent-pid", type=int, default=0,
        help="exit when this process (the Electron app) quits or crashes",
    )
    args = parser.parse_args()

    if args.parent_pid:
        def watch_parent():
            # Polls rather than blocking on a stdin pipe: on Windows a pending
            # synchronous read on stdin deadlocks later DLL loads (scipy, torch).
            # Killing the child handle isn't enough either, since python.exe
            # can be a launcher stub with the real interpreter as its child.
            import psutil
            while psutil.pid_exists(args.parent_pid):
                time.sleep(2)
            os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()

    backend = Backend(port=args.port)
    if not args.no_voice:
        backend.start_voice()
    backend.start_reminders()
    if not args.no_voice:
        backend.start_briefing()
    try:
        asyncio.run(backend.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
