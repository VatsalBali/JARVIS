"""
JARVIS - Day 1: Talk to a local LLM (Ollama) with tool calling.

SETUP (run these on YOUR machine, not in a sandbox):
1. Install Ollama: https://ollama.com/download
2. Pull a model:      ollama pull llama3.1:8b
3. Install deps:      pip install ollama
4. Run this file:     python jarvis_core.py
"""

import ollama
import json
from datetime import datetime

MODEL = "llama3.1:8b"


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

    response = ollama.chat(
        model=MODEL,
        messages=history,
        tools=TOOLS,
    )

    message = response["message"]

    # Did the model ask to call a tool?
    if message.get("tool_calls"):
        history.append(message)  # record the assistant's tool-call request

        for tool_call in message["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]

            fn = AVAILABLE_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"

            # Feed the tool's result back to the model as a "tool" message
            history.append({
                "role": "tool",
                "content": str(result),
            })

        # Ask the model to produce a final natural-language answer now that
        # it has the tool result in hand.
        followup = ollama.chat(model=MODEL, messages=history, tools=TOOLS)
        final_text = followup["message"]["content"]
        history.append({"role": "assistant", "content": final_text})
        return final_text

    # No tool needed - just a normal reply.
    history.append({"role": "assistant", "content": message["content"]})
    return message["content"]


if __name__ == "__main__":
    print("JARVIS Day 1 - type 'quit' to exit\n")
    conversation_history = []

    while True:
        user_text = input("You: ")
        if user_text.lower() in ("quit", "exit"):
            break
        reply = run_conversation(user_text, conversation_history)
        print(f"JARVIS: {reply}\n")