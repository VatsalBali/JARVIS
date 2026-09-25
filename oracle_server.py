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
    {"type":"hello","state":"asleep","muted":false}
    {"type":"state","state":"asleep|wake|listening|thinking|speaking|followup"}
    {"type":"level","value":0.0-1.0}
    {"type":"transcript","text":"...","final":false}
    {"type":"caption","text":"..."}
    {"type":"tool","name":"list_upcoming_events","label":"Checking your calendar…"}
    {"type":"confirm","id":"...","action":"...","details":"...","tier":"confirm|warn"}
    {"type":"confirm_closed","id":"...","ok":false,"reason":"timeout|cancel|answered"}
    {"type":"turn","user":"...","reply":"..."}          a finished voice turn
    {"type":"muted","value":true}
    {"type":"open_chat"}
    {"type":"result","id":"...","ok":true,"value":...}  reply to a "call"

Events from the UI:
    {"type":"confirm_reply","id":"...","ok":true}
    {"type":"cancel"}
    {"type":"open_chat"}
    {"type":"mute","value":true}
    {"type":"talk"}                                      ring click / hotkey
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
        self._voice_lock = threading.Lock()
        self._tasks = set()  # in-flight RPC tasks, kept so they aren't garbage-collected

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
        self.emit({
            "type": "confirm",
            "id": request["id"],
            "action": request["title"],
            "details": request["details"],
            "tier": request["tier"],
        })
        slot["event"].wait(CONFIRM_TIMEOUT_SEC)
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

    def start_voice_turn(self):
        """One Gemini Live exchange on a worker thread. Ignored while muted
        or while another turn is running."""
        if self.muted or not self._voice_lock.acquire(blocking=False):
            return
        threading.Thread(target=self._voice_turn, daemon=True).start()

    def _voice_turn(self):
        import gemini_voice  # imported lazily: it opens a Gemini client on import
        try:
            self.emit({"type": "state", "state": "wake"})
            user_text, reply_text = gemini_voice.voice_turn_live(
                self.session.history, self.session._ensure_conversation, on_event=self.emit
            )
            if user_text:
                self.emit({"type": "turn", "user": user_text, "reply": reply_text})
        except Exception as e:
            print(f"Voice turn failed: {e}", file=sys.stderr)
            self.emit({"type": "caption", "text": "My voice link is down at the moment."})
        finally:
            # No follow-up window yet (needs the persistent Live session).
            self.emit({"type": "state", "state": "asleep"})
            self._voice_lock.release()

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
            await ws.send(json.dumps({"type": "hello", "state": self.state, "muted": self.muted}))
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
                elif kind == "open_chat":
                    self.emit({"type": "open_chat"})
                elif kind == "mute":
                    self.muted = bool(msg.get("value"))
                    self.emit({"type": "muted", "value": self.muted})
                elif kind == "talk":
                    self.start_voice_turn()
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
    parser.add_argument(
        "--exit-with-parent", action="store_true",
        help="exit when stdin closes, i.e. when the Electron parent quits or crashes",
    )
    args = parser.parse_args()

    if args.exit_with_parent:
        def watch_parent():
            # Blocks until the parent's end of the pipe closes. Killing the
            # parent's child handle isn't enough on Windows, where python.exe
            # can be a launcher stub with the real interpreter as its child.
            sys.stdin.read()
            os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()

    try:
        asyncio.run(Backend(port=args.port).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
