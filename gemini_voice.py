"""
ORACLE - voice turn v2, via Gemini Live API.

Replaces core.listen_and_transcribe() + core.run_conversation() +
core.speak() - three separate calls (local STT, cloud LLM, local TTS) -
with one streaming speech-to-speech exchange. Native audio in, native
audio out, no separate transcription/synthesis stages.

Deliberately reuses core.py's existing machinery rather than
duplicating it:
  - core.TOOLS / core.AVAILABLE_FUNCTIONS - same ~24 tools, same
    dispatch table. Only the *schema shape* differs between Groq and
    Gemini, so tool definitions still live in exactly one place
    (core.py) and get converted here.
  - core.SYSTEM_PROMPT - same ORACLE personality for both voice and
    typed conversations.
  - core.save_message - both turns land in the same SQLite
    conversation/messages tables as the typed path, so a conversation
    started by voice looks identical in the sidebar to one started by
    typing.

Scope: this is a single-shot exchange (one ring click = one connect,
one back-and-forth, then disconnect) to match ORACLE's existing
click-to-talk model - not a standing always-on session. Gemini Live's
built-in automatic turn detection replaces the calibrated-VAD logic in
core.listen_and_transcribe(); no manual silence/threshold tuning needed
here.

Setup:
    pip install google-genai
    setx GOOGLE_API_KEY "your-key-here"   (free, no card - aistudio.google.com)

MODEL below is a preview id and Google rotates these periodically -
check https://ai.google.dev/gemini-api/docs/live for the current
live-capable model if this one starts erroring.
"""

import asyncio
import contextlib
import os
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

import core  # reuse TOOLS, AVAILABLE_FUNCTIONS, SYSTEM_PROMPT, save_message

# Live model ids rotate (README 4.3), so they're config with fallbacks: set
# ORACLE_GEMINI_LIVE_MODEL to override. If a model is gone, the next one is
# tried; the "-latest" alias is the last resort because it doesn't rotate.
# Checked 2026-09-25: both fallbacks connect, call tools and speak.
FALLBACK_MODELS = ["gemini-3.8-live", "gemini-2.5-flash-native-audio-latest"]
_working_model = None


def model_candidates() -> list:
    ordered = [os.environ.get("ORACLE_GEMINI_LIVE_MODEL"), _working_model] + FALLBACK_MODELS
    return list(dict.fromkeys(m for m in ordered if m))


def remember_working_model(model: str):
    global _working_model
    _working_model = model


MODEL = model_candidates()[0]


async def connect_live(stack, config: dict):
    """Enters a Live session on the first model that exists, via an
    AsyncExitStack so errors after connecting don't trigger a fallback."""
    last_error = None
    for model in model_candidates():
        try:
            session = await stack.enter_async_context(_client.aio.live.connect(model=model, config=config))
            remember_working_model(model)
            return session
        except Exception as e:
            if "not found" not in str(e).lower() and "not supported" not in str(e).lower():
                raise
            print(f"Live model {model} unavailable, trying the next one.")
            last_error = e
    raise last_error

INPUT_RATE = 16000    # what we send the mic at
OUTPUT_RATE = 24000    # what Gemini's audio replies come back at
CHUNK_MS = 100
CHUNK_SAMPLES = INPUT_RATE * CHUNK_MS // 1000

_client = genai.Client()  # picks up GOOGLE_API_KEY from env


def _to_gemini_tools() -> list:
    """Converts core.TOOLS (Groq/OpenAI function-calling shape) into
    Gemini's function_declarations shape. Same schemas, just without
    the {"type": "function", "function": {...}} wrapper - so every
    tool's description/parameters still only needs to be written once,
    in core.py, and both voice backends stay in sync automatically."""
    declarations = []
    for t in core.TOOLS:
        fn = t["function"]
        declarations.append({
            "name": fn["name"],
            "description": fn["description"],
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return declarations


_B64_CHARS = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n")


def pcm_bytes(data: bytes) -> bytes:
    """Audio from the Live API should arrive decoded. With pydantic older
    than google-genai needs (<2.12), it arrives still base64-encoded, and
    playing that text as PCM sounds like pure static. Real PCM is almost
    never made only of base64 characters, so decode when it is."""
    if len(data) >= 16 and len(data) % 4 == 0 and _B64_CHARS.issuperset(data):
        import base64
        try:
            return base64.b64decode(data, validate=False)
        except ValueError:
            pass
    return data


def system_instruction(text: str) -> types.Content:
    """Typed rather than a plain {"parts": [...]} dict: with older pydantic
    versions the dict gets validated as a Part and the connect fails."""
    return types.Content(parts=[types.Part(text=text)])


def voice_settings() -> tuple:
    """(voice name or None for Gemini's default, extra style instruction).
    Set with core.set_setting("voice_name", "Charon") and
    core.set_setting("voice_accent", "british"), or ORACLE_VOICE."""
    name = os.environ.get("ORACLE_VOICE") or core.get_setting("voice_name") or None
    accent = (core.get_setting("voice_accent") or "").strip().lower()
    style = " Speak with a calm, refined British accent." if accent == "british" else ""
    return name, style


def _live_config(system_text: str = None) -> dict:
    voice, style = voice_settings()
    config = {
        "response_modalities": ["AUDIO"],
        "tools": [{"function_declarations": _to_gemini_tools()}],
        # Ask Gemini to also give us text transcripts of both sides of
        # the exchange - needed so we can save/display the turn in the
        # chat log exactly like the typed path does. Without these,
        # we'd only have audio, with no text to store in SQLite or
        # show in chat.html.
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }
    if voice:
        config["speech_config"] = types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))
    # Responsiveness. Measured 2026-09-26: no thinking cuts first-audio time
    # from 0.88 s to 0.64 s on gemini-3.8-live - voice commands rarely need it
    # (set voice_thinking=on to restore). A 500 ms silence ends the owner's
    # turn instead of Gemini's longer default wait.
    if (core.get_setting("voice_thinking") or "off").lower() != "on":
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    config["realtime_input_config"] = types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
            silence_duration_ms=int(core.get_setting("voice_end_silence_ms") or 500),
        )
    )
    return config


def _level(samples: np.ndarray) -> float:
    """0-1 loudness of an int16 chunk, scaled so normal speech lands
    around 0.3-0.8 - drives the orb's core pulse."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
    return min(1.0, rms * 6.0)


async def _run_turn(on_status=None, on_event=None) -> tuple:
    """
    One full exchange over a fresh Live session: streams mic audio in,
    plays audio replies as they arrive, dispatches any tool calls via
    core.execute_tool, and returns (user_text, reply_text) - the
    transcripts of what was said on each side, for the caller to save
    and display. Ends when Gemini signals turn_complete.

    on_event(dict), if given, receives the orb events from README 4.3:
    state (listening/thinking/speaking), level, transcript and caption.
    """
    user_text_parts = []
    reply_text_parts = []
    stop_sending = asyncio.Event()
    state = {"current": None}

    def emit(event: dict):
        if on_event:
            try:
                on_event(event)
            except Exception as e:
                print(f"on_event failed: {e}")

    def set_state(new_state: str):
        if state["current"] != new_state:
            state["current"] = new_state
            emit({"type": "state", "state": new_state})

    set_state("listening")

    async with contextlib.AsyncExitStack() as stack:
        session = await connect_live(stack, _live_config())

        async def send_mic():
            stream = sd.InputStream(samplerate=INPUT_RATE, channels=1, dtype="int16")
            stream.start()
            try:
                while not stop_sending.is_set():
                    data, _ = await asyncio.to_thread(stream.read, CHUNK_SAMPLES)
                    if state["current"] == "listening":
                        emit({"type": "level", "value": _level(data)})
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=data.tobytes(),
                            mime_type=f"audio/pcm;rate={INPUT_RATE}",
                        )
                    )
            finally:
                stream.stop()
                stream.close()

        mic_task = asyncio.create_task(send_mic())

        out_stream = sd.OutputStream(samplerate=OUTPUT_RATE, channels=1, dtype="int16")
        out_stream.start()

        try:
            async for response in session.receive():
                if response.data is not None:
                    if on_status:
                        on_status("SPEAKING...")
                    set_state("speaking")
                    audio = np.frombuffer(pcm_bytes(response.data), dtype=np.int16)
                    emit({"type": "level", "value": _level(audio)})
                    await asyncio.to_thread(out_stream.write, audio)

                elif response.tool_call:
                    # Pause the mic while tools run and the model
                    # thinks - mirrors the existing turn-taking feel
                    # of the ring-click flow (listen, then think).
                    stop_sending.set()
                    if on_status:
                        on_status("THINKING...")
                    set_state("thinking")

                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        # Same dispatcher as the typed path, so risky
                        # actions get the owner's confirmation here too.
                        result = await asyncio.to_thread(
                            core.execute_tool, core.AVAILABLE_FUNCTIONS, fc.name, dict(fc.args or {})
                        )
                        function_responses.append(
                            types.FunctionResponse(
                                id=fc.id, name=fc.name, response={"result": str(result)}
                            )
                        )
                    await session.send_tool_response(function_responses=function_responses)

                    stop_sending = asyncio.Event()
                    mic_task = asyncio.create_task(send_mic())

                # Not elif: newer Live models send transcripts in the same
                # message as the audio chunk.
                if response.server_content:
                    sc = response.server_content
                    if sc.input_transcription and sc.input_transcription.text:
                        user_text_parts.append(sc.input_transcription.text)
                        emit({"type": "transcript", "text": "".join(user_text_parts).strip(), "final": False})
                    if sc.output_transcription and sc.output_transcription.text:
                        reply_text_parts.append(sc.output_transcription.text)
                        emit({"type": "caption", "text": "".join(reply_text_parts).strip()})
                    if sc.turn_complete:
                        stop_sending.set()
                        emit({"type": "transcript", "text": "".join(user_text_parts).strip(), "final": True})
                        break
        finally:
            stop_sending.set()
            mic_task.cancel()
            out_stream.stop()
            out_stream.close()

    return "".join(user_text_parts).strip(), "".join(reply_text_parts).strip()


def voice_turn_live(history: list, ensure_conversation, on_status=None, on_event=None) -> tuple:
    """
    Synchronous entry point for UI.py's threaded ring-click flow - same
    role as core.listen_and_transcribe()+run_conversation()+speak()
    combined into one streaming exchange.

    ensure_conversation: callable(user_text) -> conversation_id.
    Called once the user's transcript is known (we don't have it
    upfront, unlike the old listen-then-respond flow), so the caller
    can create/attach the SQLite conversation and update the chat UI
    at the right moment - mirrors Api._ensure_conversation.

    Returns (user_text, reply_text) so the caller can update the ring
    status / chat log the same way it already does for the Groq path.
    """
    user_text, reply_text = asyncio.run(_run_turn(on_status=on_status, on_event=on_event))

    if not user_text:
        return user_text, reply_text

    conversation_id = ensure_conversation(user_text)

    user_msg = {"role": "user", "content": user_text}
    history.append(user_msg)
    core.save_message(user_msg, conversation_id)

    if reply_text:
        assistant_msg = {"role": "assistant", "content": reply_text}
        history.append(assistant_msg)
        core.save_message(assistant_msg, conversation_id)

    return user_text, reply_text
