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
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

import core  # reuse TOOLS, AVAILABLE_FUNCTIONS, SYSTEM_PROMPT, save_message

MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"

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


def _live_config() -> dict:
    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": {"parts": [{"text": core.SYSTEM_PROMPT}]},
        "tools": [{"function_declarations": _to_gemini_tools()}],
        # Ask Gemini to also give us text transcripts of both sides of
        # the exchange - needed so we can save/display the turn in the
        # chat log exactly like the typed path does. Without these,
        # we'd only have audio, with no text to store in SQLite or
        # show in chat.html.
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }


async def _run_turn(on_status=None) -> tuple:
    """
    One full exchange over a fresh Live session: streams mic audio in,
    plays audio replies as they arrive, dispatches any tool calls via
    core.AVAILABLE_FUNCTIONS, and returns (user_text, reply_text) - the
    transcripts of what was said on each side, for the caller to save
    and display. Ends when Gemini signals turn_complete.
    """
    user_text_parts = []
    reply_text_parts = []
    stop_sending = asyncio.Event()

    async with _client.aio.live.connect(model=MODEL, config=_live_config()) as session:

        async def send_mic():
            stream = sd.InputStream(samplerate=INPUT_RATE, channels=1, dtype="int16")
            stream.start()
            try:
                while not stop_sending.is_set():
                    data, _ = await asyncio.to_thread(stream.read, CHUNK_SAMPLES)
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
                    audio = np.frombuffer(response.data, dtype=np.int16)
                    out_stream.write(audio)

                elif response.tool_call:
                    # Pause the mic while tools run and the model
                    # thinks - mirrors the existing turn-taking feel
                    # of the ring-click flow (listen, then think).
                    stop_sending.set()
                    if on_status:
                        on_status("THINKING...")

                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        fn = core.AVAILABLE_FUNCTIONS.get(fc.name)
                        try:
                            result = fn(**fc.args) if fn else f"Unknown tool: {fc.name}"
                        except Exception as e:
                            result = f"Error: {e}"
                        function_responses.append(
                            types.FunctionResponse(
                                id=fc.id, name=fc.name, response={"result": str(result)}
                            )
                        )
                    await session.send_tool_response(function_responses=function_responses)

                    stop_sending = asyncio.Event()
                    mic_task = asyncio.create_task(send_mic())

                elif response.server_content:
                    sc = response.server_content
                    if sc.input_transcription and sc.input_transcription.text:
                        user_text_parts.append(sc.input_transcription.text)
                    if sc.output_transcription and sc.output_transcription.text:
                        reply_text_parts.append(sc.output_transcription.text)
                    if sc.turn_complete:
                        stop_sending.set()
                        break
        finally:
            stop_sending.set()
            mic_task.cancel()
            out_stream.stop()
            out_stream.close()

    return "".join(user_text_parts).strip(), "".join(reply_text_parts).strip()


def voice_turn_live(history: list, ensure_conversation, on_status=None) -> tuple:
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
    user_text, reply_text = asyncio.run(_run_turn(on_status=on_status))

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
