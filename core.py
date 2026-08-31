"""
JARVIS - Day 1: Talk to an LLM (via Groq's free API) with tool calling.

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
import psutil
from datetime import datetime

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are JARVIS, a personal AI assistant in the spirit of the one from Iron Man: "
    "calm, dry-witted, quietly loyal, and understated rather than effusive. A touch of "
    "wit is welcome where it fits naturally, but never at the expense of being direct "
    "and efficient when the user actually needs something done - personality seasons "
    "your responses, it doesn't pad them out. When a question needs real information "
    "about this machine or the world, call a tool instead of guessing. To open an "
    "application (like Notepad, Chrome, or Spotify), call launch_app directly with the "
    "app name - do not use list_files or open_file to search for it first."
)

# A directory listing goes into the conversation and is re-sent on every later
# turn, so an uncapped one (System32 is ~23k tokens) blows the API rate limit.
MAX_LISTED_FILES = 50

MAX_TOOL_ROUNDS = 5

def _get_data_dir() -> str:
    """
    Returns a proper per-user, writable directory for JARVIS's persistent
    data (currently just the SQLite DB). A relative path like
    "jarvis_memory.db" only works reliably when you run `python core.py`
    by hand from this exact folder - it breaks for a packaged .exe
    launched via Windows autostart, which often runs with an unexpected
    working directory (frequently System32). %LOCALAPPDATA%\\JARVIS is
    the standard place per-user app data belongs on Windows.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.path.expanduser("~/.local/share")
    data_dir = os.path.join(base, "JARVIS")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


DB_PATH = os.path.join(_get_data_dir(), "jarvis_memory.db")

# How many of the most recent messages we actually SEND to the model each
# turn. Everything is still saved to disk - this only limits what gets
# resent on each API call, which is what keeps token usage (and cost/rate
# limits) from growing unbounded as a conversation gets long.
MAX_HISTORY_MESSAGES = 20


# ---------------------------------------------------------------------------
# PERSISTENCE: save every message to SQLite so JARVIS remembers past
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
    JARVIS picks up roughly where the last session left off."""
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
    """Extracts and returns text content from a file so JARVIS can summarize
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
    """Tool-facing wrapper: formats live system stats as text for JARVIS
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


# Lazily created for the same reason as the Tavily client - a missing or
# broken toast library shouldn't crash the whole app at startup, only the
# one tool that actually needs it.
_toaster = None


def send_notification(title: str, message: str = "") -> str:
    """
    Shows a real Windows toast notification (the kind that pops up from
    the notification area), separate from anything shown in the JARVIS
    chat window itself - useful for things JARVIS wants to flag even if
    you're not actively looking at the app.
    """
    if sys.platform != "win32":
        return "Error: toast notifications are only supported on Windows."

    global _toaster
    try:
        from windows_toasts import Toast, WindowsToaster

        if _toaster is None:
            _toaster = WindowsToaster("JARVIS")

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
                "Search the web for current information - news, facts, prices, "
                "anything that requires up-to-date or external knowledge you "
                "wouldn't already know."
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
            "name": "send_notification",
            "description": (
                "Show a real Windows toast/system notification - separate from "
                "the chat window, visible even if the user isn't looking at "
                "JARVIS right now. Use this for alerts the user should notice "
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


def generate_greeting(resuming: bool) -> str:
    """
    One-off, lightweight LLM call for a short greeting - deliberately NOT
    run through run_conversation/tools, since a greeting never needs a tool
    and this keeps startup to a single fast API call instead of a full
    tool-calling round trip.
    """
    context = (
        "This is a fresh start with no prior conversation."
        if not resuming else
        "You are resuming a conversation with the user from a previous session."
    )
    prompt = (
        f"The current time is {get_current_time()}. {context} "
        "Greet the user in one short, natural sentence, in character as established "
        "in your system prompt - calm, dry-witted, quietly loyal. Not a generic "
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

    print("JARVIS core - terminal test mode. Type 'quit' to exit.\n")
    print("(For the full JARVIS experience, run ui.py instead.)\n")

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    prior_messages = load_recent_history()
    if prior_messages:
        conversation_history.extend(prior_messages)

    greeting = generate_greeting(resuming=bool(prior_messages))
    print(f"JARVIS: {greeting}\n")
    conversation_history.append({"role": "assistant", "content": greeting})

    while True:
        user_text = input("You: ")
        if user_text.lower() in ("quit", "exit"):
            break
        reply = run_conversation(user_text, conversation_history)
        print(f"JARVIS: {reply}\n")