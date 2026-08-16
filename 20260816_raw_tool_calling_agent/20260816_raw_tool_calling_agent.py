# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0",
#     "python-dotenv>=1.0",
# ]
# ///
"""
Raw OpenAI Tool-Calling Agent
=============================

A minimal, framework-less implementation of an autonomous LLM agent with tool calling.
Uses standard OpenAI Chat Completions API via OpenRouter with plain dictionary tool definitions.

Usage:
    uv run 20260816_raw_tool_calling_agent.py "What is the system time?"
    uv run 20260816_raw_tool_calling_agent.py "List 3 files in the current directory" -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
logger = logging.getLogger("agent")


# ==============================================================================
# 1. Tool Implementation
# ==============================================================================
def run_shell(command: str) -> str:
    """Execute a shell command on host system (PowerShell on Windows, Bash on Unix)."""
    try:
        is_windows = platform.system() == "Windows"
        shell_cmd = (
            ["powershell", "-NoProfile", "-Command", command]
            if is_windows
            else ["bash", "-c", command]
        )
        res = subprocess.run(
            shell_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        out = res.stdout
        if res.stderr:
            out += f"\n[stderr]\n{res.stderr}"
        if res.returncode != 0:
            out += f"\n[exit code: {res.returncode}]"
        return out.strip() or "(no output)"
    except Exception as e:
        return f"Error executing command: {e}"


# ==============================================================================
# 2. Tool Definitions (Raw JSON Schema) & Function Map
# ==============================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command (PowerShell on Windows, Bash on Unix) and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute, e.g. 'Get-ChildItem' or 'ls'.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

TOOL_FUNCTIONS = {
    "run_shell": run_shell,
}


# ==============================================================================
# 3. Client & Agent Loop
# ==============================================================================
def get_client() -> OpenAI:
    """Get configured OpenAI client for OpenRouter."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in environment or .env file.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def run_agent(
    query: str,
    model: str = "deepseek/deepseek-v4-flash",
    max_iterations: int = 10,
) -> str:
    """
    Run reasoning and tool-calling loop for a given query, returning final response string.
    """
    client = get_client()
    messages = [{"role": "user", "content": query}]

    logger.info("Starting agent for query: %r (model: %s)", query, model)

    for iteration in range(1, max_iterations + 1):
        logger.debug("Iteration %d/%d: Requesting completion...", iteration, max_iterations)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # Exit loop when model returns final answer without tool calls
        if not msg.tool_calls:
            logger.info("Agent finished reasoning in %d iterations.", iteration)
            return msg.content or ""

        # Process tool calls and append results
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            logger.info("Tool Call: %s(%s)", fn_name, args)

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f"Tool execution failed: {e}"
            else:
                result = f"Unknown tool: {fn_name}"

            logger.debug("Tool Output (%d chars): %s", len(result), result.replace("\n", " ")[:150])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    logger.warning("Reached max iterations (%d) without final answer.", max_iterations)
    return "Error: Maximum iteration limit reached before completion."


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw OpenAI Tool-Calling Agent.")
    parser.add_argument("query", type=str, help="Query / instruction for the agent")
    parser.add_argument("--model", "-m", type=str, default="deepseek/deepseek-v4-flash", help="Model ID")
    parser.add_argument("--max-iter", type=int, default=10, help="Max tool calling iterations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    answer = run_agent(query=args.query, model=args.model, max_iterations=args.max_iter)

    print("\n--- Final Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
