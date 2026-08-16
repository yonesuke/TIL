# LLM Tool-Calling Agent from Scratch (OpenAI API / OpenRouter)

フレームワーク（LangChain や OpenAI Agents SDK など）を使用せず、素の **OpenAI Chat Completions API** と **JSON 辞書による Tool 定義** だけで自律型ツール呼び出しエージェントを構築する仕組みと実装の解説。

## 概要

LLM エージェント（Tool-Calling Agent）の核心は、**「LLM にツールの定義（JSON Schema）を渡し、LLM が返したツール実行指示をローカルで実行して、その結果を会話履歴に追加して再度 LLM に投げる」** というシンプルな推論ループにあります。

外部エージェントフレームワークを使わずに標準の `openai` ライブラリだけで書くことで、通信の内部構造やメッセージプロトコルの挙動を明確に理解・制御できます。

---

## 仕組みとプロトコル詳細

### 1. ツール定義（JSON Schema）

API に渡す `tools` 引数には、実行可能な関数のスキーマを JSON 辞書として定義します。

```python
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
```

### 2. HTTP POST リクエストボディの構造

エージェントの各ステップで OpenRouter / OpenAI API の `/chat/completions` エンドポイントへ POST されるペイロードは以下の構造です：

```json
{
  "model": "deepseek/deepseek-v4-flash",
  "messages": [ /* 会話履歴配列 */ ],
  "tools": [ /* ツール定義配列 */ ]
}
```

### 3. 会話履歴 (`messages`) の推移

エージェントが推論し、ツールを実行して最終回答に至るまでのメッセージ推移の例です：

```
[1. User Input]
  role: "user"
  content: "What is the system time? Check using shell command."
       │
       ▼ (LLM推論)
[2. Assistant Response (Tool Call)]
  role: "assistant"
  content: null
  tool_calls: [
    {
      "id": "call_abc123",
      "type": "function",
      "function": { "name": "run_shell", "arguments": "{\"command\": \"date\"}" }
    }
  ]
       │
       ▼ (ローカルで run_shell('date') を実行)
[3. Tool Output]
  role: "tool"
  tool_call_id: "call_abc123"
  content: "2026-08-17 07:45:00"
       │
       ▼ (再度LLM推論)
[4. Final Assistant Response]
  role: "assistant"
  content: "The current system time is August 17, 2026, 07:45:00."
```

#### 各ロールの役割

| `role` | 主要フィールド | 説明 |
| :--- | :--- | :--- |
| `user` | `content` | ユーザーからの指示・質問。 |
| `assistant` (通常) | `content` | LLM からの最終テキスト回答。`tool_calls` が無ければループ終了。 |
| `assistant` (ツール呼出) | `content` (`null`), `tool_calls` | LLM が要求するツール関数名と引数 JSON 文字列。各呼び出しに固有の `id` が付与される。 |
| `tool` | `tool_call_id`, `content` | Python 側で実行した結果文字列。どの `tool_calls` に対応するかを `tool_call_id` で指定する。 |

---

## 推論ループの実装

```python
def run_agent(query: str, model: str = "deepseek/deepseek-v4-flash", max_iterations: int = 10) -> str:
    client = get_client()
    messages = [{"role": "user", "content": query}]

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # ツール呼び出しが無ければ最終回答を出力して終了
        if not msg.tool_calls:
            return msg.content or ""

        # ツール呼び出しの処理
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            result = TOOL_FUNCTIONS[fn_name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Error: Maximum iteration limit reached."
```

---

## 実行方法

PEP 723 に対応しているため、`uv run` で直接実行できます（依存関係の `openai`, `python-dotenv` は自動インストールされます）。

### 環境変数の設定 (`.env`)

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
```

### 実行コマンド

```bash
# 基本実行
uv run 20260816_raw_tool_calling_agent/20260816_raw_tool_calling_agent.py "What is the system time?"

# 詳細ログ付き (-v)
uv run 20260816_raw_tool_calling_agent/20260816_raw_tool_calling_agent.py "List 3 python files in current directory" -v
```

---

## ファイル構成

| ファイル | 説明 |
| :--- | :--- |
| `20260816_raw_tool_calling_agent.py` | 素の OpenAI API を使った Tool-Calling Agent 実装 (PEP 723 対応) |
| `20260816_raw_tool_calling_agent.md` | 本解説ドキュメント |
