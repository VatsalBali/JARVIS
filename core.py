"""
ORACLE - Day 1: Talk to an LLM (via Groq's free API) with tool calling.

Why Groq and not a local model: local inference needs a real GPU. With
integrated graphics only, running even a "small" model on CPU is painfully
slow. Groq gives free, fast, hosted inference instead - and it's
OpenAI-compatible, so the code below looks almost identical to what you'd
write for OpenAI or any other compatible provider.

SETUP:
1. Get a free API key: https://console.groq.com  (no credit card needed)
2. Set it as an environment variable so it's never hardcoded in this file:
       export GROQ_API_KEY="your-key-here"      (Mac/Linux)
       setx GROQ_API_KEY "your-key-here"         (Windows, new terminal after)
3. Install deps:      pip install groq
4. Run this file:     python core.py
"""

from groq import Groq
from tavily import TavilyClient
import json
import os
import sys
import sqlite3
import subprocess
import shutil
import webbrowser
from urllib.parse import quote_plus
import psutil
import sounddevice as sd
import numpy as np
import threading
import collections
import re
from pathlib import Path
from datetime import datetime

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are ORACLE, a personal AI assistant in the spirit of JARVIS from Iron Man: "
    "calm, dry-witted, quietly loyal, and understated rather than effusive. A touch of "
    "wit is welcome where it fits naturally, but never at the expense of being direct "
    "and efficient when the user actually needs something done - personality seasons "
    "your responses, it doesn't pad them out. When a question needs real information "
    "about this machine or the world, call a tool instead of guessing. To open an "
    "application (like Notepad, Chrome, or Spotify), call launch_app directly with the "
    "app name - do not use list_files or open_file to search for it first."
)

# Name ORACLE uses when greeting you on wake-word activation.
USER_NAME = "Oracle"

# A directory listing goes into the conversation and is re-sent on every later
# turn, so an uncapped one (System32 is ~23k tokens) blows the API rate limit.
MAX_LISTED_FILES = 50

MAX_TOOL_ROUNDS = 5

def _get_data_dir() -> str:
    """
    Returns a proper per-user, writable directory for ORACLE.s persistent
    data (currently just the SQLite DB). A relative path like
    "oracle_memory.db" only works reliably when you run `python core.py`
    by hand from this exact folder - it breaks for a packaged .exe
    launched via Windows autostart, which often runs with an unexpected
    working directory (frequently System32). %LOCALAPPDATA%\\ORACLE is
    the standard place per-user app data belongs on Windows.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.path.expanduser("~/.local/share")
    data_dir = os.path.join(base, "ORACLE")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


DB_PATH = os.path.join(_get_data_dir(), "oracle_memory.db")

# How many of the most recent messages we actually SEND to the model each
# turn. Everything is still saved to disk - this only limits what gets
# resent on each API call, which is what keeps token usage (and cost/rate
# limits) from growing unbounded as a conversation gets long.
MAX_HISTORY_MESSAGES = 20


# ---------------------------------------------------------------------------
# PERSISTENCE: save every message to SQLite so ORACLE remembers past
# conversations even after the script exits and restarts. This is the
# "item #5 - conversation memory" piece leveling up from a plain in-RAM list
# to something durable.
# ---------------------------------------------------------------------------

def init_db():
    """Creates the messages table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            tool_call_id TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_message(message: dict):
    """Appends a single message to the database, exactly as it will be
    replayed into `history` later."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (role, content, tool_calls, tool_call_id, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            message.get("role"),
            message.get("content"),
            # tool_calls is a list/dict - SQLite only stores text/numbers/blobs,
            # so we serialize it to a JSON string and deserialize on load.
            json.dumps(message["tool_calls"]) if message.get("tool_calls") else None,
            message.get("tool_call_id"),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def load_recent_history(limit: int = MAX_HISTORY_MESSAGES) -> list:
    """Loads the most recent `limit` messages from disk, oldest first, so
    ORACLE picks up roughly where the last session left off."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, tool_calls, tool_call_id FROM messages "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    rows.reverse()  # we fetched newest-first; put back in chronological order

    history = []
    for role, content, tool_calls_json, tool_call_id in rows:
        msg = {"role": role, "content": content}
        if tool_calls_json:
            msg["tool_calls"] = json.loads(tool_calls_json)
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        history.append(msg)

    # Same boundary issue as trim_history: if the LIMIT cut lands right after
    # an assistant tool_calls message but before its tool response, drop the
    # orphaned leading tool message(s).
    while history and history[0]["role"] == "tool":
        history.pop(0)

    return history


# ---------------------------------------------------------------------------
# STEP 1: Define a "tool" - a real Python function the LLM can decide to call.
# This is the foundation of item #6 (basic computer commands). The LLM never
# runs code itself - it just tells us "call get_current_time with these args"
# and WE execute it and hand the result back.
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def list_files(directory: str = ".") -> str:
    """Lists files in a given directory."""
    try:
        files = os.listdir(directory)
    except Exception as e:
        return f"Error: {e}"

    if len(files) > MAX_LISTED_FILES:
        shown = json.dumps(files[:MAX_LISTED_FILES])
        return f"{shown}\n(showing {MAX_LISTED_FILES} of {len(files)} entries)"
    return json.dumps(files)


def open_file(path: str) -> str:
    """Opens a file with whatever application is set as the OS default for
    its type - e.g. a .pdf opens in your PDF viewer, a .docx in Word. This
    is exactly what happens when you double-click a file in File Explorer."""
    if not os.path.isfile(path):
        return f"Error: '{path}' does not exist or is not a file."

    try:
        if sys.platform == "win32":
            os.startfile(path)  # Windows-only; launches the default handler
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=True)
        else:
            subprocess.run(["xdg-open", path], check=True)
        return f"Opened '{path}'."
    except Exception as e:
        return f"Error opening '{path}': {e}"


def _find_start_menu_shortcut(app_name: str):
    """
    Searches Windows Start Menu shortcut folders for a .lnk file whose name
    contains app_name (case-insensitive) - this is essentially what happens
    when you press the Windows key and type an app's name. Most installed
    apps (Obsidian included) aren't on PATH at all; they only exist as a
    shortcut here, which is why a plain PATH-based launch fails for them.
    Returns the shortcut's full path if found, else None.
    """
    search_dirs = []
    appdata = os.environ.get("APPDATA")
    programdata = os.environ.get("PROGRAMDATA")
    if appdata:
        search_dirs.append(os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs"))
    if programdata:
        search_dirs.append(os.path.join(programdata, "Microsoft", "Windows", "Start Menu", "Programs"))

    name_lower = app_name.lower()
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".lnk") and name_lower in f.lower():
                    return os.path.join(root, f)
    return None


def launch_app(app_name: str) -> str:
    """
    Launches an application by name (e.g. "notepad", "chrome", "obsidian")
    WITHOUT searching the general filesystem first. Two strategies, tried
    in order:

      1. Look for a matching Start Menu shortcut (.lnk) and launch that
         directly - this covers the vast majority of installed apps, since
         that's genuinely how Windows itself finds them by name.
      2. Fall back to letting the shell resolve the name via PATH - this
         covers built-in commands like "notepad" or "calc" that don't have
         Start Menu shortcuts but ARE directly runnable.

    Unlike a naive Popen-and-forget, this actually checks whether the
    launch succeeded: a failing command (like "not recognized") normally
    exits almost immediately, so we wait briefly and inspect the result
    instead of always reporting success.
    """
    if sys.platform == "win32":
        shortcut = _find_start_menu_shortcut(app_name)
        if shortcut:
            try:
                os.startfile(shortcut)
                return f"Launched '{app_name}' (via Start Menu shortcut)."
            except Exception as e:
                return f"Error launching '{app_name}': {e}"

        # No shortcut found - fall back to PATH resolution via the shell.
        try:
            proc = subprocess.Popen(
                app_name, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                # A command that fails to resolve (like "not recognized")
                # exits almost instantly. A real GUI app keeps running, so
                # this timeout is what lets us tell the two apart without
                # blocking forever on a successful, long-running launch.
                stdout, stderr = proc.communicate(timeout=1.5)
                if proc.returncode != 0:
                    detail = stderr.strip() or stdout.strip() or f"exit code {proc.returncode}"
                    return (
                        f"Error launching '{app_name}': {detail}. "
                        f"No Start Menu shortcut matched this name either - "
                        f"try the app's exact display name."
                    )
                return f"Launched '{app_name}'."
            except subprocess.TimeoutExpired:
                # Still running past the timeout - treat as a successful,
                # ongoing GUI app launch rather than waiting indefinitely.
                return f"Launched '{app_name}'."
        except Exception as e:
            return f"Error launching '{app_name}': {e}"

    elif sys.platform == "darwin":
        try:
            subprocess.run(["open", "-a", app_name], check=True)
            return f"Launched '{app_name}'."
        except Exception as e:
            return f"Error launching '{app_name}': {e}"
    else:
        try:
            subprocess.Popen([app_name])
            return f"Launched '{app_name}'."
        except Exception as e:
            return f"Error launching '{app_name}': {e}"


# How many characters of extracted file text we hand to the model at once.
# Same reasoning as MAX_LISTED_FILES: a 40-page PDF's full text would blow
# past the context window and get resent on every later turn.
MAX_FILE_CHARS = 6000


def read_file(path: str) -> str:
    """Extracts and returns text content from a file so ORACLE can summarize
    it or answer questions about it. Supports .txt/.md, .pdf, and .docx -
    the format is picked automatically from the file extension."""
    if not os.path.isfile(path):
        return f"Error: '{path}' does not exist or is not a file."

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in (".txt", ".md", ".csv", ".json", ".log"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif ext == ".docx":
            from docx import Document
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)

        else:
            return f"Error: unsupported file type '{ext}'. Supported: .txt, .md, .csv, .json, .log, .pdf, .docx"

    except Exception as e:
        return f"Error reading '{path}': {e}"

    if not text.strip():
        return f"'{path}' appears to be empty or its text couldn't be extracted (e.g. a scanned/image-only PDF)."

    if len(text) > MAX_FILE_CHARS:
        return f"{text[:MAX_FILE_CHARS]}\n\n(truncated - showing first {MAX_FILE_CHARS} of {len(text)} characters)"
    return text


def move_file(source: str, destination: str) -> str:
    """Moves or renames a file. If destination is a directory, the file is
    moved into it keeping its original name; otherwise destination is
    treated as the new full path/name."""
    if not os.path.isfile(source):
        return f"Error: source '{source}' does not exist or is not a file."
    try:
        shutil.move(source, destination)
        return f"Moved '{source}' to '{destination}'."
    except Exception as e:
        return f"Error moving '{source}' to '{destination}': {e}"


def delete_file(path: str) -> str:
    """Permanently deletes a file. There is no undo - this does not use the
    Recycle Bin, it removes the file directly."""
    if not os.path.isfile(path):
        return f"Error: '{path}' does not exist or is not a file."
    try:
        os.remove(path)
        return f"Deleted '{path}'."
    except Exception as e:
        return f"Error deleting '{path}': {e}"


def create_file(path: str, content: str = "") -> str:
    """Creates a new text file with the given content. Fails if the file
    already exists, to avoid silently overwriting something."""
    if os.path.exists(path):
        return f"Error: '{path}' already exists. Use a different name or delete it first."
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Created '{path}'."
    except Exception as e:
        return f"Error creating '{path}': {e}"


def get_system_stats_dict() -> dict:
    """
    Returns live system stats as structured data (not a string) - used
    directly by the UI for the sidebar's live bars, and also as the source
    of truth behind the get_system_info tool below. Keeping this separate
    from the tool-facing text version means the UI can poll it every few
    seconds without spending any LLM tokens or API calls.
    """
    cpu_percent = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk_path = "C:\\" if sys.platform == "win32" else "/"
    disk = psutil.disk_usage(disk_path)

    stats = {
        "cpu_percent": round(cpu_percent, 1),
        "ram_percent": round(mem.percent, 1),
        "ram_used_gb": round(mem.used / (1024 ** 3), 1),
        "ram_total_gb": round(mem.total / (1024 ** 3), 1),
        "disk_percent": round(disk.percent, 1),
        "disk_used_gb": round(disk.used / (1024 ** 3), 1),
        "disk_total_gb": round(disk.total / (1024 ** 3), 1),
        "gpu": None,
    }

    # VRAM: only meaningful for a dedicated GPU. GPUtil supports NVIDIA
    # cards; on integrated-graphics machines (or without GPUtil/no NVIDIA
    # driver) this will fail, and we deliberately leave "gpu" as None
    # rather than fabricate a number.
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            stats["gpu"] = {
                "name": g.name,
                "vram_percent": round(g.memoryUtil * 100, 1),
                "vram_used_mb": round(g.memoryUsed),
                "vram_total_mb": round(g.memoryTotal),
            }
    except Exception:
        pass

    return stats


def get_system_info() -> str:
    """Tool-facing wrapper: formats live system stats as text for ORACLE
    to read out or discuss in conversation."""
    s = get_system_stats_dict()
    lines = [
        f"CPU usage: {s['cpu_percent']}%",
        f"RAM usage: {s['ram_percent']}% ({s['ram_used_gb']} GB of {s['ram_total_gb']} GB)",
        f"Storage usage: {s['disk_percent']}% ({s['disk_used_gb']} GB of {s['disk_total_gb']} GB)",
    ]
    if s["gpu"]:
        g = s["gpu"]
        lines.append(
            f"GPU: {g['name']} - VRAM {g['vram_percent']}% "
            f"({g['vram_used_mb']} MB of {g['vram_total_mb']} MB)"
        )
    else:
        lines.append("GPU/VRAM: no dedicated GPU detected (integrated graphics).")
    return "\n".join(lines)


# Lazily created so a missing TAVILY_API_KEY doesn't crash the whole app on
# startup - it only becomes a problem the moment web_search is actually
# called, with a clear error message instead of a startup traceback.
_tavily_client = None


def web_search(query: str) -> str:
    """Searches the web via Tavily and returns a handful of results with
    titles, URLs, and short content summaries."""
    global _tavily_client

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=api_key)

    try:
        response = _tavily_client.search(query=query, max_results=5)
    except Exception as e:
        return f"Error performing web search: {e}"

    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = (r.get("content") or "")[:400]
        formatted.append(f"{title}\n{url}\n{content}")

    return "\n\n".join(formatted)


def open_web_search(query: str) -> str:
    """
    Opens the user's default web browser to a real, visible search results
    page - unlike web_search above, which fetches results as text for
    ORACLE to read and summarize, this actually launches a browser tab so
    the user can look at and interact with the results themselves.
    """
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    try:
        webbrowser.open(url, new=2)  # new=2: open in a new tab, not the same window
        return f"Opened a web search for '{query}' in your browser."
    except Exception as e:
        return f"Error opening web search: {e}"


def open_url(url: str) -> str:
    """
    Opens a specific web address in the user's default browser - for when
    the user names an actual site to go to (e.g. "open youtube.com"),
    as opposed to searching for something (see open_web_search/web_search).
    """
    url = url.strip()

    # Add a scheme if the user just said "youtube.com" without one.
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = f"https://{url}"

    # Basic sanity check - must look like an actual domain, and only
    # http(s) is allowed (guards against something like a javascript: or
    # file: URI slipping through, which "open this website" should never
    # trigger).
    if not re.match(r"^https?://[^\s]+\.[^\s]+", url, re.IGNORECASE):
        return f"Error: '{url}' doesn't look like a valid web address."

    try:
        webbrowser.open(url, new=2)
        return f"Opened {url}"
    except Exception as e:
        return f"Error opening '{url}': {e}"


# ---------------------------------------------------------------------------
# VOICE INPUT: local, free speech-to-text via faster-whisper. Recording
# happens entirely in Python (via sounddevice) rather than in the browser -
# this keeps audio capture and transcription in one place instead of
# encoding audio in JS and shipping it across the pywebview bridge.
# ---------------------------------------------------------------------------

# "base" balances speed and accuracy reasonably well on CPU-only hardware
# (no dedicated GPU here). "tiny" is faster but noticeably less accurate;
# "small"/"medium" are more accurate but slower per transcription.
WHISPER_MODEL_SIZE = "base"
SAMPLE_RATE = 16000  # Whisper's native input rate - recording at this rate
                      # directly avoids a separate resampling step.

_whisper_model = None
_recording_stream = None
_recording_frames = []


def _get_whisper_model():
    """Lazily loads the Whisper model on first use (this can take a few
    seconds and downloads model weights on first run ever) rather than
    slowing down every ORACLE startup, most of which won't use voice."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def start_recording() -> str:
    """Begins capturing microphone audio into memory. Call
    stop_recording_and_transcribe() to end it and get the transcribed text."""
    global _recording_stream, _recording_frames

    if _recording_stream is not None:
        return "Already recording."

    _recording_frames = []

    def _callback(indata, frames, time_info, status):
        _recording_frames.append(indata.copy())

    try:
        _recording_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_callback
        )
        _recording_stream.start()
        return "Recording started."
    except Exception as e:
        _recording_stream = None
        return f"Error starting recording: {e}"


def stop_recording_and_transcribe() -> str:
    """Stops capturing audio and transcribes whatever was recorded."""
    global _recording_stream, _recording_frames

    if _recording_stream is None:
        return "Error: not currently recording."

    try:
        _recording_stream.stop()
        _recording_stream.close()
    except Exception as e:
        _recording_stream = None
        return f"Error stopping recording: {e}"

    _recording_stream = None

    if not _recording_frames:
        return "Error: no audio was captured."

    audio = np.concatenate(_recording_frames, axis=0).flatten()
    _recording_frames = []

    # Rough silence check - a very quiet/empty recording (e.g. mic muted,
    # or the stop button hit almost immediately) shouldn't be sent to
    # Whisper at all, since it tends to hallucinate text from near-silence.
    if np.abs(audio).mean() < 0.001:
        return "Error: no speech detected (audio was silent)."

    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(audio, language=None)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as e:
        return f"Error transcribing audio: {e}"

    return text if text else "Error: could not understand the audio."


# Tunable VAD (voice activity detection) parameters for listen_and_transcribe.
CALIBRATION_DURATION_SEC = 0.5   # brief ambient-noise measurement before listening starts
SPEECH_THRESHOLD_MULTIPLIER = 3.5  # how far above the measured noise floor counts as speech
MIN_SPEECH_THRESHOLD = 0.01      # floor, so a near-silent room doesn't make the mic's own
                                  # self-noise register as "speech"
SILENCE_DURATION_SEC = 1.2       # pause length (after speech) that ends listening
NO_SPEECH_TIMEOUT_SEC = 8.0      # give up if nothing is said within this long (calibration
                                  # eats into this window, so it's a bit longer than before)
MAX_RECORDING_SEC = 20.0         # hard safety cap regardless of VAD state


def listen_and_transcribe() -> str:
    """
    Listens via the microphone until you pause after speaking (or a safety
    timeout elapses), then transcribes what was captured - a single
    blocking call, unlike start_recording/stop_recording_and_transcribe's
    manual two-step. This is what the global hotkey uses: press it, talk,
    stop talking, and this returns once it detects you're done - the same
    turn-taking style used by voice mode in other AI chat apps.

    The first CALIBRATION_DURATION_SEC of audio is used to measure YOUR
    mic's actual ambient noise floor rather than assuming a fixed volume
    level - different microphones and rooms have very different baseline
    noise, so a single hardcoded threshold either misses quiet speech or
    (more often) never sees true silence at all if the room's baseline is
    louder than the hardcoded guess.
    """
    frames = []
    done = threading.Event()
    state = {
        "speech_detected": False,
        "silence_frames": 0,
        "total_frames": 0,
        "calibration_rms": [],
        "threshold": None,  # set once calibration finishes
    }

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        state["total_frames"] += frame_count
        rms = float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
        elapsed = state["total_frames"] / SAMPLE_RATE

        # Phase 1: calibration - just measure, don't judge speech/silence yet.
        if state["threshold"] is None:
            state["calibration_rms"].append(rms)
            if elapsed >= CALIBRATION_DURATION_SEC:
                noise_floor = float(np.mean(state["calibration_rms"]))
                state["threshold"] = max(
                    noise_floor * SPEECH_THRESHOLD_MULTIPLIER, MIN_SPEECH_THRESHOLD
                )
            return

        # Phase 2: normal VAD logic, using the calibrated threshold.
        if rms > state["threshold"]:
            state["speech_detected"] = True
            state["silence_frames"] = 0
        elif state["speech_detected"]:
            state["silence_frames"] += frame_count

        silence_elapsed = state["silence_frames"] / SAMPLE_RATE

        if state["speech_detected"] and silence_elapsed >= SILENCE_DURATION_SEC:
            raise sd.CallbackStop()
        if not state["speech_detected"] and elapsed >= NO_SPEECH_TIMEOUT_SEC:
            raise sd.CallbackStop()
        if elapsed >= MAX_RECORDING_SEC:
            raise sd.CallbackStop()

    def _finished():
        done.set()

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),  # ~100ms chunks for responsive VAD
            callback=callback,
            finished_callback=_finished,
        )
    except Exception as e:
        return f"Error starting microphone: {e}"

    with stream:
        # Extra couple seconds of margin beyond MAX_RECORDING_SEC, purely
        # so this wait() can't itself hang forever in some edge case.
        done.wait(timeout=MAX_RECORDING_SEC + 2)

    if not frames:
        return "Error: no audio was captured."

    audio = np.concatenate(frames, axis=0).flatten()

    if not state["speech_detected"]:
        return "Error: no speech detected."

    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(audio, language=None)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as e:
        return f"Error transcribing audio: {e}"

    return text if text else "Error: could not understand the audio."


# ---------------------------------------------------------------------------
# WAKE WORD: no free pretrained wake-word engine (openWakeWord, Porcupine,
# etc.) ships with an arbitrary custom word like "oracle" - those need a
# real training pipeline with audio data, which isn't practical to set up
# here. Instead, this continuously transcribes short rolling windows of
# audio with our existing Whisper model and checks the text for the word.
#
# Honest trade-off: this is meaningfully more CPU-hungry running
# continuously than a dedicated lightweight wake-word model would be
# (those typically use <1% CPU; ours periodically runs real speech
# recognition). Using the smallest "tiny" Whisper model specifically for
# this constant background check - separate from the more accurate "base"
# model used for actual commands - keeps that cost as low as reasonably
# possible while still working for a custom word with zero training.
# ---------------------------------------------------------------------------

WAKE_WORD = "oracle"
WAKE_WHISPER_MODEL_SIZE = "tiny"
WAKE_WINDOW_SEC = 3.0          # how much recent audio is checked each time
WAKE_CHECK_INTERVAL_SEC = 1.5  # how often to check (creates overlap with the
                                # window above, so the word isn't missed if
                                # it lands right on a boundary)
WAKE_CHUNK_SEC = 0.5           # size of each buffered audio chunk

_wake_whisper_model = None
_wake_buffer = collections.deque(maxlen=int(WAKE_WINDOW_SEC / WAKE_CHUNK_SEC))
_wake_buffer_lock = threading.Lock()
_wake_stream = None


def _get_wake_whisper_model():
    """Separate, smaller model instance from the one used for actual
    command transcription - see the module-level comment above for why."""
    global _wake_whisper_model
    if _wake_whisper_model is None:
        from faster_whisper import WhisperModel
        _wake_whisper_model = WhisperModel(WAKE_WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _wake_whisper_model


def _wake_callback(indata, frame_count, time_info, status):
    with _wake_buffer_lock:
        _wake_buffer.append(indata.copy())


def start_wake_word_stream() -> str:
    """Starts the continuous background microphone stream that feeds the
    rolling buffer check_for_wake_word() reads from."""
    global _wake_stream
    if _wake_stream is not None:
        return "Already listening for wake word."
    try:
        _wake_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * WAKE_CHUNK_SEC),
            callback=_wake_callback,
        )
        _wake_stream.start()
        return "Wake word listening started."
    except Exception as e:
        _wake_stream = None
        return f"Error starting wake word stream: {e}"


def stop_wake_word_stream():
    """Stops the wake-word background stream - used while a voice turn is
    actively being handled, so the mic isn't fought over by two streams
    at once and so the buffer doesn't re-trigger on residual audio."""
    global _wake_stream
    if _wake_stream is not None:
        try:
            _wake_stream.stop()
            _wake_stream.close()
        except Exception:
            pass
        _wake_stream = None
    with _wake_buffer_lock:
        _wake_buffer.clear()


def check_for_wake_word() -> bool:
    """
    Transcribes whatever's currently in the rolling buffer and checks for
    the wake word. Called periodically (see run_wake_word_listener in
    ui.py) rather than continuously, since running Whisper truly
    non-stop would be excessive - the WAKE_CHECK_INTERVAL_SEC gap plus
    the buffer's overlap is the balance between catching the word
    promptly and not pegging the CPU constantly.
    """
    with _wake_buffer_lock:
        if not _wake_buffer:
            return False
        audio = np.concatenate(list(_wake_buffer), axis=0).flatten()

    # Cheap check before bothering Whisper at all - skip near-silent windows.
    if np.abs(audio).mean() < 0.001:
        return False

    try:
        model = _get_wake_whisper_model()
        segments, _ = model.transcribe(audio, language=None)
        text = " ".join(segment.text for segment in segments).lower()
    except Exception:
        return False

    cleaned = re.sub(r"[^a-z\s]", "", text)
    return WAKE_WORD in cleaned


# ---------------------------------------------------------------------------
# TEXT-TO-SPEECH: local, free voice output via Piper. Every ORACLE reply
# gets spoken, whether it came from typing or voice - see ui.py for where
# this actually gets called (kept out of the conversational logic here so
# core.py doesn't need to know whether a reply is about to hit a chat
# window, a voice turn, or both).
# ---------------------------------------------------------------------------

PIPER_VOICE_NAME = "en_US-lessac-medium"  # solid general-purpose default voice
_piper_voice = None


def _strip_markdown_for_speech(text: str) -> str:
    """
    Removes markdown formatting before handing text to Piper - without
    this, a reply like "**Note:** run `python ui.py`" would be read aloud
    literally as "asterisk asterisk Note asterisk asterisk colon run
    backtick python ui dot py backtick", which is unusable. Code blocks
    are dropped entirely rather than read aloud, since spoken code is
    rarely useful and often very long.
    """
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _get_piper_voice():
    """
    Lazily loads (and downloads, on first-ever use) the Piper voice model.
    Downloaded once into ORACLE's own data directory - after that first
    run, this is instant and fully offline.
    """
    global _piper_voice
    if _piper_voice is None:
        from piper import PiperVoice
        from piper.download_voices import download_voice

        voice_dir = Path(_get_data_dir()) / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)

        download_voice(PIPER_VOICE_NAME, voice_dir)  # no-ops if already downloaded

        model_path = voice_dir / f"{PIPER_VOICE_NAME}.onnx"
        config_path = voice_dir / f"{PIPER_VOICE_NAME}.onnx.json"
        _piper_voice = PiperVoice.load(str(model_path), str(config_path))
    return _piper_voice


def speak(text: str) -> str:
    """
    Synthesizes text to speech locally via Piper and plays it through the
    default audio output. Blocks until playback finishes - callers that
    don't want to wait (e.g. so a chat message can appear immediately
    instead of only after the audio finishes) should run this in a
    background thread instead of calling it directly.
    """
    cleaned = _strip_markdown_for_speech(text)
    if not cleaned:
        return "Error: no speakable text after removing formatting."

    try:
        voice = _get_piper_voice()
        chunks = list(voice.synthesize(cleaned))
        if not chunks:
            return "Error: no audio was generated."

        audio = np.concatenate([c.audio_float_array for c in chunks])
        sample_rate = chunks[0].sample_rate

        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        return "Spoken."
    except Exception as e:
        return f"Error speaking: {e}"


# Lazily created for the same reason as the Tavily client - a missing or
# broken toast library shouldn't crash the whole app at startup, only the
# one tool that actually needs it.
_toaster = None


def send_notification(title: str, message: str = "") -> str:
    """
    Shows a real Windows toast notification (the kind that pops up from
    the notification area), separate from anything shown in the ORACLE
    chat window itself - useful for things ORACLE wants to flag even if
    you're not actively looking at the app.
    """
    if sys.platform != "win32":
        return "Error: toast notifications are only supported on Windows."

    global _toaster
    try:
        from windows_toasts import Toast, WindowsToaster

        if _toaster is None:
            _toaster = WindowsToaster("ORACLE")

        toast = Toast()
        toast.text_fields = [title, message] if message else [title]
        _toaster.show_toast(toast)
        return f"Notification sent: '{title}'"
    except Exception as e:
        return f"Error sending notification: {e}"


# This is the "menu" we hand to the model, describing each tool so it knows
# when and how to call it. This schema format is the same shape used by
# OpenAI, Anthropic, and Ollama - it's becoming a de facto standard.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory on the local machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory to list. Defaults to current directory.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": (
                "Open a specific FILE (e.g. PDF, DOCX, image, text file) using the "
                "operating system's default application for that file type - "
                "the same as double-clicking it in a file browser. For launching "
                "an APPLICATION by name (e.g. Notepad, Chrome, Spotify) instead "
                "of opening a file, use launch_app instead - it's much faster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full or relative path to the file to open.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": (
                "Launch an application by name (e.g. 'notepad', 'chrome', 'spotify', "
                "'calculator'). Use this for opening PROGRAMS - it's a single fast "
                "call. Do NOT use list_files or open_file to hunt for an application's "
                "executable file first; launch_app resolves common app names directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the application to launch, e.g. 'notepad' or 'chrome'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read and extract the text content of a file so you can summarize it "
                "or answer questions about it. Supports .txt, .md, .csv, .json, .log, "
                ".pdf, and .docx."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full or relative path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file on the local machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Current path of the file."},
                    "destination": {"type": "string", "description": "New path, name, or destination directory."},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Permanently delete a file from the local machine. This cannot be undone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full or relative path to the file to delete."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new text file with the given content on the local machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full or relative path for the new file."},
                    "content": {"type": "string", "description": "Text content to write into the file. Defaults to empty."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get current system resource usage: CPU, RAM, disk/storage, and GPU/VRAM if available.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information and get results back as text "
                "for you to read and summarize in the conversation - the user does NOT "
                "see a browser. Use this when the user is asking a question you need "
                "an answer to. For when the user wants to actually SEE search results "
                "themselves in their browser, use open_web_search instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_web_search",
            "description": (
                "Opens the user's default web browser to a real search results page "
                "they can see and interact with - use this when the user asks you to "
                "search for something and clearly wants to look at the results "
                "themselves (e.g. 'search for X', 'look up Y for me'), as opposed to "
                "asking you a question you should just answer directly (use web_search "
                "for that instead). If the user instead names a specific website to "
                "visit (e.g. 'open youtube.com', 'go to github'), use open_url instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Opens a specific website in the user's default browser - use this "
                "when the user names an actual site or address to go to (e.g. 'open "
                "youtube.com', 'go to github.com/anthropics'), as opposed to searching "
                "for something (use web_search or open_web_search for that instead)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The web address to open, e.g. 'youtube.com' or 'https://github.com'.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": (
                "Show a real Windows toast/system notification - separate from "
                "the chat window, visible even if the user isn't looking at "
                "ORACLE right now. Use this for alerts the user should notice "
                "even if they're not in the app."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short notification title."},
                    "message": {"type": "string", "description": "Notification body text. Optional."},
                },
                "required": ["title"],
            },
        },
    },
]

# Map tool names to actual Python functions so we can execute them by name.
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "list_files": list_files,
    "open_file": open_file,
    "launch_app": launch_app,
    "read_file": read_file,
    "move_file": move_file,
    "delete_file": delete_file,
    "create_file": create_file,
    "get_system_info": get_system_info,
    "web_search": web_search,
    "open_web_search": open_web_search,
    "open_url": open_url,
    "send_notification": send_notification,
}


def trim_history(history: list) -> list:
    """
    Keeps the system prompt plus only the most recent MAX_HISTORY_MESSAGES
    messages. Everything is still saved in full to SQLite - this only
    controls what gets sent to the API each turn, since token usage (and
    Groq's rate limits) scale with how much history you resend every call.
    """
    system_msgs = [m for m in history if m["role"] == "system"]
    other_msgs = [m for m in history if m["role"] != "system"]
    trimmed = other_msgs[-MAX_HISTORY_MESSAGES:]

    # A "tool" message only makes sense immediately after the assistant
    # message that requested it. If the cut landed between them, the API
    # will reject the orphaned tool message - so drop leading tool messages
    # until we start on a clean boundary.
    while trimmed and trimmed[0]["role"] == "tool":
        trimmed.pop(0)

    return system_msgs + trimmed


def run_conversation(user_input: str, history: list) -> str:
    """
    Sends the user's message + history to the model. The model may need several
    rounds of tool calls - each result can prompt the next call - so we keep
    going until it returns a plain text answer.
    """
    history[:] = trim_history(history)
    user_msg = {"role": "user", "content": user_input}
    history.append(user_msg)
    save_message(user_msg)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
        )

        message = response.choices[0].message

        # No tool needed - just a normal reply.
        if not message.tool_calls:
            final_msg = {"role": "assistant", "content": message.content}
            history.append(final_msg)
            save_message(final_msg)
            return message.content

        # Record the assistant's tool-call request in history. We build this
        # dict manually rather than using message.model_dump() - the full
        # dump includes extra fields (like "annotations") that the API is
        # happy to SEND but refuses to RECEIVE back. Only include what the
        # spec actually requires.
        tool_call_msg = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }
        history.append(tool_call_msg)
        save_message(tool_call_msg)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            # Arguments come back as a JSON string, not a dict - must parse.
            fn_args = json.loads(tool_call.function.arguments)

            fn = AVAILABLE_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"

            # Feed the tool's result back to the model as a "tool" message.
            # tool_call_id links this result to the specific call above -
            # required when a model requests multiple tool calls at once.
            tool_result_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
            history.append(tool_result_msg)
            save_message(tool_result_msg)

    give_up = f"I kept calling tools without reaching an answer ({MAX_TOOL_ROUNDS} rounds). Try asking a narrower question."
    give_up_msg = {"role": "assistant", "content": give_up}
    history.append(give_up_msg)
    save_message(give_up_msg)
    return give_up


def generate_wake_greeting() -> str:
    """
    One-off, lightweight LLM call for a short greeting spoken when the
    wake word activates ORACLE - deliberately NOT run through
    run_conversation/tools, since a greeting never needs a tool and this
    keeps activation to a single fast API call instead of a full
    tool-calling round trip. Explicitly asked to vary its phrasing each
    time rather than reusing the same line on every activation.
    """
    prompt = (
        f"The current time is {get_current_time()}. The user (whom you address as "
        f"'{USER_NAME}') just activated you with your wake word. Greet them by name "
        "in one short, natural sentence, in character as established in your system "
        "prompt - calm, dry-witted, quietly loyal. Vary your phrasing meaningfully "
        "each time rather than repeating a fixed template - not a generic "
        "'How can I help you today?'"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # cp1252, the default Windows console encoding, can't represent characters
    # that routinely appear in replies (curly quotes, U+202F).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    init_db()

    print("ORACLE core - terminal test mode. Type 'quit' to exit.\n")
    print("(For the full ORACLE experience, run ui.py instead.)\n")

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    prior_messages = load_recent_history()
    if prior_messages:
        conversation_history.extend(prior_messages)

    while True:
        user_text = input("You: ")
        if user_text.lower() in ("quit", "exit"):
            break
        reply = run_conversation(user_text, conversation_history)
        print(f"ORACLE: {reply}\n")