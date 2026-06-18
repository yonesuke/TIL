# SQL Agent with OpenAI Agents SDK

OpenAI Agents SDK を使った自然言語 → SQL エージェント。OpenRouter + DeepSeek で動作。

## アーキテクチャ

```
chinook_schema.toml  ──▶  DB_SCHEMA (メモリ上データ辞書)
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
 sql_db_search_tables   sql_db_schema             sql_db_query
 (キーワード検索)        (テーブル定義+サンプル)     (実DB実行)
                              │
                              ▼
                   sql_db_query_checker
                   (サブAgent as Tool)
```

## Agent Orchestration パターン

**Agents as Tools**: メインの SQL Agent が会話をコントロールしつつ、SQL レビューという限定的なサブタスクを専用サブエージェント (`sql_db_query_checker`) に委譲。`Agent.as_tool()` で実装。

## データ辞書 (TOML)

`chinook_schema.toml` がカタログの単一の情報源。テーブル説明・カラム型・外部キー・サンプル行をすべて含む。起動時に `tomllib` でロード、DB 接続不要でスキーマ探索が可能。

## 動的テーブル発見

システムプロンプトに全テーブルを列挙せず、`sql_db_search_tables(keyword)` でキーワード検索。300テーブル規模でもプロンプトが膨らまない。

## 使い方

```bash
# .env に OPENROUTER_API_KEY を設定
uv run 20260618_sql_agent_openai_sdk.py
```

## 依存

- `openai-agents` (OpenAI Agents SDK)
- `openai` (OpenRouter の OpenAI-compatible API 経由)
- `python-dotenv`, `requests`

## 参考

- [LangChain SQL Agent](https://docs.langchain.com/oss/python/langchain/sql-agent)
- [OpenAI Agents SDK — Agent Orchestration](https://openai.github.io/openai-agents-python/agent_orchestration/)
- [Chinook Sample Database](https://github.com/lerocha/chinook-database)
