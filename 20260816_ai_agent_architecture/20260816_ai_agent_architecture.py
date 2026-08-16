# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
# ]
# ///
"""
AI Agent Minimal Architecture Demo (2-While-Loop Pattern)

エージェントの最小限の制御構造（2重Whileループ ＋ Tool Calling）を
素のOpenAI API（または互換API）を用いて実装したリファレンスコード。

Usage:
    uv run 20260816_ai_agent_architecture/20260816_ai_agent_architecture.py
"""

import json
import os
import subprocess
from typing import Any, Dict, List
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. ツール定義 (JSON Schema)
# -----------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "ターミナルコマンドを実行して標準出力/標準エラー出力を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "実行するシェルコマンド (例: 'ls -la', 'python --version')"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Pythonの数式を計算して結果を返す",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "計算する数式 (例: '2 ** 10 + 42')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# -----------------------------------------------------------------------------
# 2. ローカルツール実行関数
# -----------------------------------------------------------------------------
def execute_local_function(name: str, arguments: Dict[str, Any]) -> str:
    """エージェントから要求された関数をローカル環境で安全に実行して文字列で返す"""
    if name == "run_bash_command":
        cmd = arguments.get("command", "")
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            output = res.stdout if res.returncode == 0 else f"[Error (code {res.returncode})]: {res.stderr}"
            return output.strip() or "(No output)"
        except Exception as e:
            return f"Command execution error: {e}"

    elif name == "calculate":
        expr = arguments.get("expression", "")
        try:
            # 簡易電卓 (実務ではAST等で安全に評価)
            return str(eval(expr, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Calculation error: {e}"

    return f"Unknown tool: {name}"

# -----------------------------------------------------------------------------
# 3. エージェントの2重Whileループ
# -----------------------------------------------------------------------------
def run_agent_loop():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[!] OPENAI_API_KEY is not set. Running in dry-run explanation mode.")
        print("    Set OPENAI_API_KEY to test interactive agent execution.")
        return

    client = OpenAI()
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "あなたは優秀なエンジニアリングエージェントです。"
                "与えられた目標を達成するために、適切なツールを呼び出し、"
                "ツールの実行結果を観察しながら自律的にステップを進めてください。"
            )
        }
    ]

    print("🤖 Agent CLI Ready (Type 'exit' to quit)\n" + "=" * 50)

    # 【第1のループ】：ユーザーとの対話を受け付ける外側ループ
    while True:
        try:
            user_input = input("\nUser > ")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip() or user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        # 【第2のループ】：ツール実行と推論を繰り返す内側ループ（エージェントループ）
        while True:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS_SCHEMA
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump())

            # ツール呼び出し（tool_calls）がなければタスク完了 $\rightarrow$ ループ脱出
            if not msg.tool_calls:
                print(f"\nAgent > {msg.content}")
                break

            # ツール呼び出しがあれば、ローカル関数を実行して結果を履歴に添加
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"  ⚡ [Tool Call] {func_name}({func_args})")

                tool_output = execute_local_function(func_name, func_args)
                print(f"  ↳ [Tool Result] {tool_output[:100]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_output)
                })

if __name__ == "__main__":
    run_agent_loop()
