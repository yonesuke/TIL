import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-v4-flash"


def say_hello(name: str = "") -> str:
    """A simple tool that returns a greeting message."""
    return f"Hello {name}!" if name else "Hello from OpenRouter!"


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # ダミー実装。実際はAPI叩くなど
    return f"The weather in {location} is sunny and 22°C."


# Tool定義（APIに送る用）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "say_hello",
            "description": "Returns a greeting message. Optionally takes a name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name to greet",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo'",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

# 実際の関数を名前で引けるように
TOOL_REGISTRY = {
    "say_hello": say_hello,
    "get_weather": get_weather,
}


def chat():
    client = openai.Client(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_URL,
    )
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("You>> ")
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            # pyrefly: ignore [no-matching-overload]
            response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_DEFINITIONS)

            msg = response.choices[0].message
            messages.append(msg)

            if msg.content:
                print(f"Assistant>> {msg.content}")

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name in TOOL_REGISTRY:
                    tool_func = TOOL_REGISTRY[tool_name]
                    result = tool_func(**tool_args)
                    messages.append({"role": "tool", "name": tool_name, "content": result})
                else:
                    messages.append({"role": "tool", "name": tool_name, "content": f"Tool {tool_name} not found."})

if __name__ == "__main__":
    chat()
