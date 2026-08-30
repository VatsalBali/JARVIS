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
4. Run this file:     python jarvis_core.py
"""

from groq import Groq
import json
import os
from datetime import datetime

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


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
    import os
    try:
        files = os.listdir(directory)
        return json.dumps(files)
    except Exception as e:
        return f"Error: {e}"


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
]

# Map tool names to actual Python functions so we can execute them by name.
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "list_files": list_files,
}


def run_conversation(user_input: str, history: list) -> str:
    """
    Sends the user's message + history to the model. If the model wants to
    call a tool, we run it and send the result back for a final answer.
    """
    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=TOOLS,
    )

    message = response.choices[0].message

    # Did the model ask to call a tool?
    if message.tool_calls:
        # Record the assistant's tool-call request in history. We build this
        # dict manually rather than using message.model_dump() - the full
        # dump includes extra fields (like "annotations") that the API is
        # happy to SEND but refuses to RECEIVE back. Only include what the
        # spec actually requires.
        history.append({
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
        })

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            # Arguments come back as a JSON string, not a dict - must parse.
            fn_args = json.loads(tool_call.function.arguments)

            fn = AVAILABLE_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"

            # Feed the tool's result back to the model as a "tool" message.
            # tool_call_id links this result to the specific call above -
            # required when a model requests multiple tool calls at once.
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

        # Ask the model to produce a final natural-language answer now that
        # it has the tool result in hand.
        followup = client.chat.completions.create(model=MODEL, messages=history, tools=TOOLS)
        final_text = followup.choices[0].message.content
        history.append({"role": "assistant", "content": final_text})
        return final_text

    # No tool needed - just a normal reply.
    history.append({"role": "assistant", "content": message.content})
    return message.content


if __name__ == "__main__":
    print("JARVIS Day 1 - type 'quit' to exit\n")
    conversation_history = []

    while True:
        user_text = input("You: ")
        if user_text.lower() in ("quit", "exit"):
            break
        reply = run_conversation(user_text, conversation_history)
        print(f"JARVIS: {reply}\n")