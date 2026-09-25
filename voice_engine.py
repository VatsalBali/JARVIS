"""
ORACLE - voice engine (README section 4): one microphone, the wake word,
voice identification, and Gemini Live conversations that stay open for
follow-ups and can be interrupted.

    Mic         one 16 kHz mono input stream. Every consumer (wake word, the
                Live session, voice ID, spoken yes/no, enrolment) subscribes
                to its frames; the device is never opened twice.
    WakeWord    openWakeWord on 80 ms frames. Runs while asleep, and while
                ORACLE is thinking/speaking so "Oracle" can barge in.
    VoiceID     Resemblyzer speaker embedding vs. the owner's enrolled voiceprint.
    Conversation one Gemini Live session: listening -> thinking -> speaking ->
                follow-up (~6 s, no wake word needed) -> ... -> asleep.

Nothing is sent to the cloud before a wake: until then, audio only goes to
the local wake-word model.

The wake-word model is config, not code. Until a custom "Oracle" model is
trained (docs/TRAIN_WAKE_WORD.md), the pretrained "hey jarvis" model stands
in. Drop oracle.onnx into %LOCALAPPDATA%\\ORACLE\\wakeword\\ and it's used
automatically, or set ORACLE_WAKE_MODEL to a model path or name.
"""

import asyncio
import contextlib
import csv
import os
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
from google.genai import types

import core
import gemini_voice

RATE = 16000
FRAME = 1280                      # 80 ms, openWakeWord's native frame size
FRAMES_PER_SEC = RATE // FRAME
OUTPUT_RATE = 24000               # Gemini Live audio replies

WAKE_THRESHOLD = 0.5              # per-frame score to count as a hit
WAKE_PATIENCE = 2                 # consecutive hits needed (cuts false wakes)
WAKE_COOLDOWN_SEC = 2.0
NO_SPEECH_SEC = 8.0               # give up if nothing is said after a wake
NO_REPLY_SEC = 15.0               # give up if Gemini doesn't answer after speech stops
LOCAL_VAD_FLOOR = 0.004           # speech if frame RMS > max(0.01, 3x this)
FOLLOWUP_SEC = 6.0                # README 4.1: keep listening ~6 s after a reply
STOP_DEBOUNCE_SEC = 0.7
DUCK_LEVEL = 0.3                  # other apps' volume while ORACLE talks
VOICE_ID_THRESHOLD = 0.75         # cosine similarity; tune via the voice_id_threshold setting
ENROL_SPEECH_SEC = 25.0
ENROL_TIMEOUT_SEC = 90.0

STOP_PHRASES = re.compile(
    r"^\W*((hey\s+)?(oracle|jarvis)\W*)?(stop|cancel|never\s*mind|that'?s all|that will be all)\W*$",
    re.IGNORECASE,
)
YES_WORDS = re.compile(r"\b(yes|yeah|yep|confirm|confirmed|do it|go ahead|send it|approved?)\b", re.IGNORECASE)
NO_WORDS = re.compile(r"\b(no|nope|cancel|don'?t|stop|never\s*mind)\b", re.IGNORECASE)

ENROL_PASSAGE = (
    "Read this aloud, then keep talking about anything: "
    "“Oracle, this is my voice. I work late, I drink too much coffee, "
    "and I'd like my calendar kept honest.”"
)

VOICE_STYLE = (
    " You are speaking aloud through the orb, so keep replies brief and "
    "conversational, with no markdown, lists or links."
)


def _data_dir() -> str:
    return core._get_data_dir()


def _level(samples: np.ndarray) -> float:
    return gemini_voice._level(samples)


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------

class Mic:
    """The single input stream. Frames (int16, FRAME samples) are copied to
    every subscriber queue and kept in a short history for pre-roll/voice ID."""

    def __init__(self):
        self._subs = set()
        self._lock = threading.Lock()
        self.history = deque(maxlen=3 * FRAMES_PER_SEC)
        self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self):
        if self._stream:
            return
        self._stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME, callback=self._callback
        )
        self._stream.start()

    def stop(self):
        stream, self._stream = self._stream, None
        if stream:
            stream.stop()
            stream.close()
        self.history.clear()

    def _callback(self, indata, frames, time_info, status):
        frame = indata[:, 0].copy()
        self.history.append(frame)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass

    def subscribe(self, maxsize: int = 0) -> queue.Queue:
        q = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._subs.discard(q)

    def recent(self, seconds: float) -> np.ndarray:
        frames = list(self.history)[-max(1, int(seconds * FRAMES_PER_SEC)):]
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------

def _resolve_wake_model() -> tuple:
    """(model path or pretrained name, phrase shown in the UI)."""
    override = os.environ.get("ORACLE_WAKE_MODEL") or core.get_setting("wake_model")
    if override:
        name = os.path.splitext(os.path.basename(override))[0].replace("_", " ")
        return override, name.split(" v")[0].title()
    custom = os.path.join(_data_dir(), "wakeword", "oracle.onnx")
    if os.path.isfile(custom):
        return custom, "Oracle"
    return "hey_jarvis", "Hey Jarvis"


class WakeWord:
    def __init__(self, mic: Mic, should_listen, on_wake):
        self.mic = mic
        self.should_listen = should_listen
        self.on_wake = on_wake
        self.model_ref, self.phrase = _resolve_wake_model()
        self.threshold = float(core.get_setting("wake_threshold") or WAKE_THRESHOLD)
        self._model = None
        self._thread = None
        self._running = False
        self._log_path = os.path.join(_data_dir(), "wake_log.csv")

    def start(self):
        if self._running:
            return
        from openwakeword.model import Model
        self._model = Model(wakeword_models=[self.model_ref], inference_framework="onnx")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="wakeword")
        self._thread.start()

    def stop(self):
        self._running = False

    def _log(self, kind: str, score: float):
        """Wake events and near misses, for tuning the threshold (README 4.3)."""
        try:
            new = not os.path.exists(self._log_path)
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "event", "score", "model", "threshold"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), kind, f"{score:.3f}", self.model_ref, self.threshold])
        except OSError:
            pass

    def _loop(self):
        q = self.mic.subscribe(maxsize=50)
        streak = 0
        peak = 0.0
        cooldown_until = 0.0
        was_listening = True
        try:
            while self._running:
                try:
                    frame = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                listening = self.should_listen()
                if not listening:
                    was_listening = False
                    streak = 0
                    continue
                if not was_listening:
                    self._model.reset()  # stale audio from before the pause
                    was_listening = True

                score = max(self._model.predict(frame).values())
                if score >= self.threshold:
                    streak += 1
                    peak = max(peak, score)
                else:
                    if 0 < streak < WAKE_PATIENCE and peak > 0:
                        self._log("near_miss", peak)
                    streak, peak = 0, 0.0

                now = time.monotonic()
                if streak >= WAKE_PATIENCE and now >= cooldown_until:
                    self._log("wake", peak)
                    streak, peak = 0, 0.0
                    cooldown_until = now + WAKE_COOLDOWN_SEC
                    self._model.reset()
                    self.on_wake()
        finally:
            self.mic.unsubscribe(q)


# ---------------------------------------------------------------------------
# Voice identification
# ---------------------------------------------------------------------------

class VoiceID:
    """Owner's voiceprint, stored locally only (README 4.3)."""

    def __init__(self):
        self.path = os.path.join(_data_dir(), "voiceprint.npy")
        self._encoder = None
        self._lock = threading.Lock()
        self.print = np.load(self.path) if os.path.isfile(self.path) else None

    @property
    def enrolled(self) -> bool:
        return self.print is not None

    @property
    def threshold(self) -> float:
        return float(core.get_setting("voice_id_threshold") or VOICE_ID_THRESHOLD)

    def _embed(self, pcm: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._encoder is None:
                from resemblyzer import VoiceEncoder
                self._encoder = VoiceEncoder("cpu", verbose=False)
            wav = pcm.astype(np.float32) / 32768.0
            return self._encoder.embed_utterance(wav)

    def similarity(self, pcm: np.ndarray):
        if not self.enrolled or pcm.size < RATE // 2:
            return None
        emb = self._embed(pcm)
        return float(np.dot(emb, self.print) / (np.linalg.norm(emb) * np.linalg.norm(self.print)))

    def verify(self, pcm: np.ndarray) -> tuple:
        """(verified, score). Not enrolled or too little audio -> not verified."""
        score = self.similarity(pcm)
        return (score is not None and score >= self.threshold), score

    def enroll(self, pcm: np.ndarray):
        emb = self._embed(pcm)
        np.save(self.path, emb)
        self.print = emb


def _voiced(frame: np.ndarray, floor: float) -> bool:
    rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
    return rms > max(0.01, floor * 3.0)


# ---------------------------------------------------------------------------
# Audio out + ducking
# ---------------------------------------------------------------------------

class Player:
    """Callback-driven output so a barge-in can drop queued audio at once."""

    def __init__(self):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.level = 0.0
        self._stream = sd.OutputStream(
            samplerate=OUTPUT_RATE, channels=1, dtype="int16", callback=self._callback
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        need = frames * 2
        with self._lock:
            chunk = bytes(self._buf[:need])
            del self._buf[:need]
        samples = np.frombuffer(chunk, dtype=np.int16)
        self.level = _level(samples) if samples.size else 0.0
        out = np.zeros(frames, dtype=np.int16)
        out[: samples.size] = samples
        outdata[:, 0] = out

    def write(self, data: bytes):
        with self._lock:
            self._buf.extend(data)

    def flush(self):
        with self._lock:
            self._buf.clear()

    @property
    def busy(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def close(self):
        self._stream.stop()
        self._stream.close()


class Ducker:
    """Lowers other apps' volume while ORACLE listens and speaks (pycaw)."""

    def __init__(self):
        self._saved = {}

    def duck(self):
        if self._saved:
            return
        try:
            from pycaw.pycaw import AudioUtilities
            me = os.getpid()
            for s in AudioUtilities.GetAllSessions():
                if s.Process and s.Process.pid != me:
                    vol = s.SimpleAudioVolume
                    level = vol.GetMasterVolume()
                    self._saved[s.Process.pid] = (vol, level)
                    vol.SetMasterVolume(level * DUCK_LEVEL, None)
        except Exception as e:
            print(f"Ducking failed: {e}")

    def restore(self):
        saved, self._saved = self._saved, {}
        for vol, level in saved.values():
            try:
                vol.SetMasterVolume(level, None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# One Live conversation
# ---------------------------------------------------------------------------

class Conversation:
    """A Gemini Live session that lasts across follow-ups. Runs its own
    asyncio loop on a worker thread; other threads talk to it via
    barge_in() and end()."""

    def __init__(self, engine, wake_audio: np.ndarray):
        self.engine = engine
        self.emit = engine.emit
        self.mic_q = engine.mic.subscribe(maxsize=200)
        self.wake_audio = wake_audio
        self.loop = None
        self.done = None
        self.mode = "listening"       # listening | thinking | speaking | draining | followup
        self.sending = True           # forward mic audio to Gemini
        self.drop_audio = False       # discard model audio after a barge-in
        self.barged = False           # the in-progress turn was cut off by the owner
        self.heard_speech = False
        self.listen_started = time.monotonic()
        self.last_voice_at = self.listen_started
        self.last_server_at = self.listen_started
        self.tool_running = False
        self.followup_until = 0.0
        self.last_transcript_at = 0.0
        self.user_parts = []
        self.reply_parts = []
        self.turn_audio = []          # this turn's user audio, for voice ID
        self._verified = None         # cached per turn
        self.player = None

    # ---- called from other threads ----

    def barge_in(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self._barge_in)

    def end(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self._end)

    @property
    def speaker_verified_now(self) -> bool:
        return bool(self._verified)

    # ---- helpers (loop thread unless noted) ----

    def _set_mode(self, mode: str, state: str = None):
        self.mode = mode
        self.emit({"type": "state", "state": state or mode})

    def _end(self):
        if self.done and not self.done.is_set():
            self.done.set()

    def _barge_in(self):
        if self.mode not in ("thinking", "speaking", "draining"):
            return
        self.player.flush()
        # Only a turn still in progress will send a turn_complete to skip.
        self.barged = self.mode in ("speaking", "thinking")
        self.drop_audio = self.barged
        self._start_listening()

    def _start_listening(self):
        self.sending = True
        self.heard_speech = False
        self.listen_started = time.monotonic()
        self._set_mode("listening")

    def speaker_verified(self) -> bool:
        """Called on the tool thread. Wake word + this turn's speech vs. the
        owner's voiceprint; computed once per turn."""
        if self._verified is None:
            parts = ([self.wake_audio] if self.wake_audio.size else []) + list(self.turn_audio)
            pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
            ok, score = self.engine.voice_id.verify(pcm)
            self._verified = ok
            self.emit({
                "type": "speaker", "verified": ok, "enrolled": self.engine.voice_id.enrolled,
                "score": None if score is None else round(score, 3),
            })
        return self._verified

    def _finish_turn(self, final: bool = False) -> bool:
        """Saves the exchange once it has a reply. After a tool call Gemini
        ends one turn silently and answers in the next, so a question with
        no reply yet is held (unless the session is ending). Returns whether
        a reply was saved."""
        user_text = "".join(self.user_parts).strip()
        reply_text = "".join(self.reply_parts).strip()
        if user_text and not reply_text and not final:
            return False
        self.user_parts, self.reply_parts = [], []
        self.turn_audio = []
        self._verified = None
        if user_text:
            self.emit({"type": "transcript", "text": user_text, "final": True})
            self.engine.save_turn(user_text, reply_text)
        return bool(reply_text)

    # ---- tasks ----

    async def _send_loop(self, session):
        while not self.done.is_set():
            try:
                frame = await asyncio.to_thread(self.mic_q.get, True, 0.2)
            except queue.Empty:
                continue
            if self.mode in ("listening", "followup"):
                self.turn_audio.append(frame)
                # Local speech detection: Gemini's transcript only arrives
                # once the owner stops talking, too late for the timeouts.
                if _voiced(frame, LOCAL_VAD_FLOOR):
                    self.last_voice_at = time.monotonic()
                    if self.mode == "listening":
                        self.heard_speech = True
                    else:
                        self.followup_until = max(self.followup_until, self.last_voice_at + 2.0)
                if self.mode == "listening":
                    self.emit({"type": "level", "value": _level(frame)})
            if self.sending:
                await session.send_realtime_input(
                    audio=types.Blob(data=frame.tobytes(), mime_type=f"audio/pcm;rate={RATE}")
                )

    async def _watchdog(self):
        while not self.done.is_set():
            await asyncio.sleep(0.1)
            now = time.monotonic()
            if self.mode == "speaking":
                self.emit({"type": "level", "value": self.player.level})
            if self.mode == "listening" and not self.heard_speech and now - self.listen_started > NO_SPEECH_SEC:
                self._end()
            elif self.mode == "listening" and self.heard_speech and now - self.last_voice_at > NO_REPLY_SEC:
                self._end()  # spoke, then silence, and Gemini never answered
            elif (self.mode == "thinking" and not self.tool_running
                    and now - self.last_server_at > NO_REPLY_SEC):
                self._end()  # waiting on an answer that never came
            elif self.mode == "draining" and not self.player.busy:
                # Reply finished playing: listen for a follow-up without the
                # wake word. Starting only now keeps ORACLE from hearing itself.
                self.sending = True
                self.turn_audio = []
                self.heard_speech = False
                self.followup_until = now + FOLLOWUP_SEC
                self._set_mode("followup")
                self.engine.ducker.restore()
            elif self.mode == "followup" and not self.heard_speech and now > self.followup_until:
                self._end()
            if (self.user_parts and now - self.last_transcript_at > STOP_DEBOUNCE_SEC
                    and STOP_PHRASES.match("".join(self.user_parts).strip())):
                self.emit({"type": "caption", "text": "Very good."})
                self._end()

    async def _receive_loop(self, session):
        fns = dict(core.AVAILABLE_FUNCTIONS)
        while not self.done.is_set():
            got_any = False
            async for response in session.receive():
                got_any = True
                self.last_server_at = time.monotonic()
                if response.data is not None and not self.drop_audio:
                    if self.mode != "speaking":
                        self.sending = False  # no mic to Gemini while it talks (no echo cancel)
                        self._set_mode("speaking")
                        self.engine.ducker.duck()
                    self.player.write(response.data)

                elif response.tool_call:
                    self.sending = False
                    self._set_mode("thinking")
                    responses = []
                    self.tool_running = True  # may wait minutes on a confirmation
                    try:
                        for fc in response.tool_call.function_calls:
                            result = await asyncio.to_thread(
                                core.execute_tool, fns, fc.name, dict(fc.args or {}), self.speaker_verified
                            )
                            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": str(result)}))
                    finally:
                        self.tool_running = False
                        self.last_server_at = time.monotonic()
                    await session.send_tool_response(function_responses=responses)

                # Not elif: newer Live models send transcripts in the same
                # message as the audio chunk.
                if response.server_content:
                    sc = response.server_content
                    if sc.interrupted:
                        self.player.flush()
                        self.drop_audio = False
                    if sc.input_transcription and sc.input_transcription.text:
                        self.user_parts.append(sc.input_transcription.text)
                        self.last_transcript_at = time.monotonic()
                        self.heard_speech = True
                        if self.mode == "followup":
                            self.listen_started = time.monotonic()
                            self._set_mode("listening")
                        self.emit({"type": "transcript", "text": "".join(self.user_parts).strip(), "final": False})
                    if sc.output_transcription and sc.output_transcription.text:
                        self.reply_parts.append(sc.output_transcription.text)
                        if not self.drop_audio:
                            self.emit({"type": "caption", "text": "".join(self.reply_parts).strip()})
                    if sc.turn_complete:
                        self.drop_audio = False
                        if self.barged:
                            # The turn that ended is the one we cut off: save
                            # it as it stands and keep listening, since the
                            # owner is already talking again.
                            self._finish_turn(final=True)
                            self.barged = False
                            continue
                        replied = self._finish_turn()
                        if not replied and self.user_parts:
                            # Silent turn after a tool call: the answer is
                            # still coming, so don't open the follow-up yet.
                            self._set_mode("thinking")
                        else:
                            # Let the reply finish playing, then the watchdog
                            # opens the follow-up window.
                            self.sending = False
                            self._set_mode("draining", state="speaking")
            if not got_any:
                self._end()  # connection closed

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.done = asyncio.Event()
        self.player = Player()
        config = gemini_voice._live_config(core.SYSTEM_PROMPT + VOICE_STYLE)
        self.engine.ducker.duck()
        self._set_mode("listening")
        try:
            async with contextlib.AsyncExitStack() as stack:
                session = await gemini_voice.connect_live(stack, config)
                tasks = [
                    asyncio.create_task(self._send_loop(session)),
                    asyncio.create_task(self._receive_loop(session)),
                    asyncio.create_task(self._watchdog()),
                ]
                await self.done.wait()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._finish_turn(final=True)
            self.engine.mic.unsubscribe(self.mic_q)
            self.player.close()
            self.engine.ducker.restore()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VoiceEngine:
    def __init__(self, emit, chat_session):
        self.emit = emit
        self.chat = chat_session
        self.mic = Mic()
        self.voice_id = VoiceID()
        self.ducker = Ducker()
        self.wake = WakeWord(self.mic, self._wake_should_listen, self._on_wake)
        self.muted = False
        self.conversation = None
        self._busy = threading.Lock()   # one conversation or enrolment at a time
        self._enrolling = False
        self._confirm_pending = False

    @property
    def wake_phrase(self) -> str:
        return self.wake.phrase

    def start(self):
        try:
            self.mic.start()
            self.wake.start()
        except Exception as e:
            print(f"Voice engine could not start: {e}")
            self.emit({"type": "caption", "text": "I can't reach the microphone."})

    def _wake_should_listen(self) -> bool:
        if self.muted or self._enrolling or self._confirm_pending:
            return False
        conv = self.conversation
        # Asleep, or mid-reply (barge-in). Not while already listening.
        return conv is None or conv.mode in ("thinking", "speaking", "draining")

    def _on_wake(self):
        conv = self.conversation
        if conv is not None:
            conv.barge_in()
            return
        self._start_conversation(self.mic.recent(1.5))

    def trigger(self):
        """Hotkey / tray / ring click: same as saying the wake word."""
        if self.muted:
            return
        conv = self.conversation
        if conv is not None:
            conv.barge_in()
        else:
            self._start_conversation(np.zeros(0, dtype=np.int16))

    def _start_conversation(self, wake_audio: np.ndarray):
        if not self.mic.running or not self._busy.acquire(blocking=False):
            return
        self.emit({"type": "state", "state": "wake"})
        conv = Conversation(self, wake_audio)
        self.conversation = conv

        def run():
            try:
                asyncio.run(conv.run())
            except Exception as e:
                print(f"Voice conversation failed: {e}")
                self.emit({"type": "caption", "text": "My voice link is down at the moment."})
            finally:
                self.conversation = None
                self._busy.release()
                self.emit({"type": "state", "state": "asleep"})

        threading.Thread(target=run, daemon=True, name="conversation").start()

    def cancel(self):
        conv = self.conversation
        if conv is not None:
            conv.end()

    def set_muted(self, muted: bool):
        self.muted = muted
        if muted:
            self.cancel()
            self.mic.stop()   # mute means the device is closed, not just ignored
        else:
            self.start()

    def save_turn(self, user_text: str, reply_text: str):
        conversation_id = self.chat._ensure_conversation(user_text)
        for msg in ({"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply_text} if reply_text else None):
            if msg:
                self.chat.history.append(msg)
                core.save_message(msg, conversation_id)
        self.emit({"type": "turn", "user": user_text, "reply": reply_text})

    # ---- spoken confirmation (README 4.1) ----

    def spoken_confirmation_allowed(self, request: dict) -> bool:
        """Spoken yes/no only inside a voice conversation whose speaker was
        verified as the owner, and never for an unrecognised voice's request."""
        conv = self.conversation
        return conv is not None and conv.speaker_verified_now and not request.get("guest")

    def listen_yes_no(self, stop: threading.Event, answer):
        """Listens for one short utterance at a time, transcribes it with the
        local Whisper model, and calls answer(True/False) on a clear yes/no.
        The answer must also come from the owner's voice."""
        self._confirm_pending = True
        q = self.mic.subscribe(maxsize=200)
        try:
            floor = 0.005
            utterance, silence = [], 0
            while not stop.is_set():
                try:
                    frame = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if _voiced(frame, floor):
                    utterance.append(frame)
                    silence = 0
                elif utterance:
                    silence += 1
                    utterance.append(frame)
                    if silence >= 6 or len(utterance) > 5 * FRAMES_PER_SEC:  # ~0.5 s pause or 5 s cap
                        pcm = np.concatenate(utterance)
                        utterance, silence = [], 0
                        verdict = self._yes_no(pcm)
                        if verdict is not None:
                            answer(verdict)
                            return
                else:
                    rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
                    floor = 0.95 * floor + 0.05 * rms
        finally:
            self.mic.unsubscribe(q)
            self._confirm_pending = False

    def _yes_no(self, pcm: np.ndarray):
        if pcm.size < RATE // 4:
            return None
        segments, _ = core._get_whisper_model().transcribe(pcm.astype(np.float32) / 32768.0, language="en")
        text = " ".join(s.text for s in segments).strip()
        if not text:
            return None
        self.emit({"type": "transcript", "text": text, "final": True})
        verdict = False if NO_WORDS.search(text) else (True if YES_WORDS.search(text) else None)
        if verdict:
            ok, _ = self.voice_id.verify(pcm)
            if not ok and self.voice_id.enrolled:
                return None  # a short "yes" is hard to verify; fall back to the click
        return verdict

    # ---- enrolment ----

    def start_enrollment(self):
        if self.muted or not self.mic.running or not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._enroll, daemon=True, name="enrol").start()
        return True

    def _enroll(self):
        self._enrolling = True
        q = self.mic.subscribe(maxsize=400)
        try:
            self.emit({"type": "state", "state": "listening", "label": "LEARNING YOUR VOICE"})
            self.emit({"type": "caption", "text": ENROL_PASSAGE})
            floor, voiced = 0.005, []
            deadline = time.monotonic() + ENROL_TIMEOUT_SEC
            while time.monotonic() < deadline and len(voiced) < ENROL_SPEECH_SEC * FRAMES_PER_SEC:
                try:
                    frame = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                self.emit({"type": "level", "value": _level(frame)})
                if _voiced(frame, floor):
                    voiced.append(frame)
                else:
                    rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
                    floor = 0.95 * floor + 0.05 * rms
            if len(voiced) < 10 * FRAMES_PER_SEC:
                self.emit({"type": "caption", "text": "I didn't catch enough of your voice. Let's try again later."})
                self.emit({"type": "enroll", "ok": False})
                time.sleep(3)
                return
            self.emit({"type": "state", "state": "thinking", "label": "LEARNING YOUR VOICE"})
            self.voice_id.enroll(np.concatenate(voiced))
            self.emit({"type": "caption", "text": "Got it. I'll know your voice from now on, V."})
            self.emit({"type": "enroll", "ok": True})
            time.sleep(3)
        except Exception as e:
            print(f"Enrolment failed: {e}")
            self.emit({"type": "caption", "text": "Something went wrong while learning your voice."})
            time.sleep(3)
        finally:
            self.mic.unsubscribe(q)
            self._enrolling = False
            self._busy.release()
            self.emit({"type": "state", "state": "asleep"})
