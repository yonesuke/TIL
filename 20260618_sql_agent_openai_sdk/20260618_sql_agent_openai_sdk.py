# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "openai-agents>=0.0.7",
#     "openai>=1.0",
#     "python-dotenv>=1.0",
#     "requests>=2.0",
# ]
# ///

"""
SQL Agent using OpenAI Agents SDK + OpenRouter + DeepSeek
==========================================================

Natural-language SQL query agent for SQLite built on the OpenAI Agents SDK.
The agent answers questions about the Chinook sample database.

**Agent Orchestration — "Agents as Tools":**
    A specialist "Query Checker" sub-agent is invoked via ``Agent.as_tool()`` to
    review every SQL query before execution. The main SQL Agent keeps control of
    the conversation while delegating query review to a focused sub-agent.

**Data Dictionary — pre-built schema cache:**
    The DB is scanned *once* at startup via PRAGMA queries.  Table definitions,
    column types, foreign keys, sample rows, and business descriptions are
    cached in memory.  ``sql_db_list_tables`` and ``sql_db_schema`` read from
    this cache — zero DB I/O per tool call.  Only ``sql_db_query`` hits the
    actual database.

**Provider:** OpenRouter (OpenAI-compatible API)
**Model:** ``deepseek/deepseek-v4-flash`` (lightweight, cost-effective)

References:
    - LangChain SQL agent: https://docs.langchain.com/oss/python/langchain/sql-agent
    - OpenAI Agents SDK orchestration: https://openai.github.io/openai-agents-python/agent_orchestration/

Usage:
    uv run 20260618_sql_agent_openai_sdk.py
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sqlite3
import textwrap
import tomllib
from collections.abc import Sequence
from typing import TypedDict

import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv()

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in environment or .env file.\n"
        "Get one at https://openrouter.ai/keys"
    )

MODEL: str = "deepseek/deepseek-v4-flash"
DB_URL: str = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DB_PATH: pathlib.Path = pathlib.Path("Chinook.db")
SCHEMA_TOML: pathlib.Path = pathlib.Path("chinook_schema.toml")

# ═══════════════════════════════════════════════════════════════════════════════
# OpenRouter Client (OpenAI-compatible) + Model
# ═══════════════════════════════════════════════════════════════════════════════

set_tracing_disabled(True)

_async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-OpenRouter-Title": "SQL Agent Demo",
    },
)

_openrouter_model = OpenAIChatCompletionsModel(
    model=MODEL,
    openai_client=_async_client,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Dictionary — loaded from TOML catalog file at startup
# ═══════════════════════════════════════════════════════════════════════════════


class ColumnInfo(TypedDict):
    name: str
    type: str
    pk: bool
    nullable: bool
    description: str


class FKInfo(TypedDict):
    from_col: str
    to_table: str
    to_col: str


class TableInfo(TypedDict, total=False):
    description: str
    row_count: int
    columns: list[ColumnInfo]
    foreign_keys: list[FKInfo]
    sample_rows: list[tuple]


DB_SCHEMA: dict[str, TableInfo] = {}


def load_schema_catalog(path: pathlib.Path) -> dict[str, TableInfo]:
    """Load table/column metadata from a TOML catalog file.

    The TOML file is the single source of truth — hand-curated with business
    descriptions that cannot be derived from DDL alone.  No PRAGMA queries,
    no DB access needed for schema lookups.

    TOML structure per table::

        [TableName]
        description = "..."
        row_count = 999

        [[TableName.columns]]
        name = "ColName"
        type = "INTEGER"
        pk = true
        nullable = false
        description = "What this column means"

        [[TableName.foreign_keys]]
        from_col = "ColName"
        to_table = "OtherTable"
        to_col = "OtherCol"
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    schema: dict[str, TableInfo] = {}
    for table_name, table_data in raw.items():
        if not isinstance(table_data, dict):
            continue

        # TOML represents [[array-of-tables]] as a list of dicts under the key
        columns_raw: list[dict] = table_data.get("columns", [])
        fks_raw: list[dict] = table_data.get("foreign_keys", [])

        schema[table_name] = {
            "description": str(table_data.get("description", "")),
            "row_count": int(table_data.get("row_count", 0)),
            "columns": [
                {
                    "name": str(c["name"]),
                    "type": str(c["type"]),
                    "pk": bool(c.get("pk", False)),
                    "nullable": bool(c.get("nullable", True)),
                    "description": str(c.get("description", "")),
                }
                for c in columns_raw
            ],
            "foreign_keys": [
                {
                    "from_col": str(fk["from_col"]),
                    "to_table": str(fk["to_table"]),
                    "to_col": str(fk["to_col"]),
                }
                for fk in fks_raw
            ],
            "sample_rows": [
                tuple(None if v == "" else v for v in sr["values"])
                for sr in table_data.get("sample_rows", [])
            ],
        }

    return schema


def _format_schema(table: str, info: TableInfo) -> str:
    """Render one table's schema as human-readable text for the LLM."""
    lines: list[str] = []

    # Table header
    desc = f"  -- {info['description']}" if info["description"] else ""
    lines.append(f"-- Table: {table} ({info['row_count']} rows){desc}")

    # Columns
    col_header = f"-- Columns ({len(info['columns'])}):"
    col_lines: list[str] = [col_header]
    for c in info["columns"]:
        flags: list[str] = []
        if c["pk"]:
            flags.append("PK")
        if not c["nullable"]:
            flags.append("NOT NULL")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        desc_str = f"  -- {c['description']}" if c.get("description") else ""
        col_lines.append(f"--   {c['name']}: {c['type']}{flag_str}{desc_str}")
    lines.append("\n".join(col_lines))

    # Foreign keys
    if info["foreign_keys"]:
        fk_lines = ["-- Foreign keys:"]
        for fk in info["foreign_keys"]:
            fk_lines.append(f"--   {fk['from_col']} -> {fk['to_table']}({fk['to_col']})")
        lines.append("\n".join(fk_lines))

    # Random sample rows (populated at startup from live DB)
    samples = info.get("sample_rows", [])
    if samples:
        col_names = [c["name"] for c in info["columns"]]
        sample_lines = ["-- Random sample:", "/*", "\t".join(col_names)]
        for row in samples:
            sample_lines.append("\t".join("NULL" if x is None else str(x) for x in row))
        sample_lines.append("*/")
        lines.append("\n".join(sample_lines))

    return "\n".join(lines)


def _search_tables(query: str, top_k: int = 10) -> str:
    """Keyword search over table descriptions and column names.

    Splits the query into keywords, scores each table by how many keywords
    match its description, column names, and column descriptions.  Returns
    the top-K results with scores.
    """
    keywords = [kw.lower() for kw in query.split() if len(kw) >= 2]
    if not keywords:
        return "No search keywords found"

    scored: list[tuple[int, str, TableInfo]] = []
    for name, info in DB_SCHEMA.items():
        # Build a searchable text blob for this table
        text = (
            f"{name} "
            f"{info.get('description', '')} "
            + " ".join(c["name"] for c in info["columns"]) + " "
            + " ".join(c.get("description", "") for c in info["columns"])
        ).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, name, info))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    lines = [f"Found {len(top)} matching table(s) for '{query}':\n"]
    for score, name, info in top:
        desc = f" — {info['description']}" if info.get("description") else ""
        lines.append(f"  {name} ({info['row_count']} rows, score={score}){desc}")
    return "\n".join(lines)


def _get_db() -> sqlite3.Connection:
    """Return a new SQLite connection to the Chinook database."""
    return sqlite3.connect(str(DB_PATH))


def download_chinook_db() -> None:
    """Download the Chinook sample database if not already present."""
    if DB_PATH.exists():
        print(f"[OK] {DB_PATH} already exists ({DB_PATH.stat().st_size:,} bytes)")
        return
    print("Downloading Chinook database ...", end=" ", flush=True)
    resp = requests.get(DB_URL, timeout=60)
    resp.raise_for_status()
    DB_PATH.write_bytes(resp.content)
    print(f"done ({len(resp.content):,} bytes)")


# ═══════════════════════════════════════════════════════════════════════════════
# SQL Tools — read from the in-memory data dictionary
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool
def sql_db_list_tables() -> str:
    """List all user tables in the database with row counts and descriptions.

    Call this to discover what tables are available before querying schemas.
    """
    if not DB_SCHEMA:
        return "Error: schema dictionary not built yet"
    lines = []
    for name in sorted(DB_SCHEMA):
        info = DB_SCHEMA[name]
        desc = f" — {info['description']}" if info["description"] else ""
        lines.append(f"{name} ({info['row_count']} rows){desc}")
    return "\n".join(lines)


@function_tool
def sql_db_search_tables(query: str) -> str:
    """Search for relevant tables by keyword or natural-language description.

    Use this as the **primary way** to find which tables are relevant to a
    question.  It searches table descriptions, column names, and column
    descriptions.  Returns top matches with relevance scores.

    Example input: ``"customer invoices"`` → returns Customer, Invoice, InvoiceLine.
    """
    if not DB_SCHEMA:
        return "Error: schema dictionary not built yet"
    return _search_tables(query)


@function_tool
def sql_db_schema(table_names: str) -> str:
    """Return full schema for one or more tables: DDL, columns, foreign keys,
    and up to 3 sample rows.

    Input: comma-separated table names, e.g. ``"Track, Genre"``.
    Call ``sql_db_list_tables`` first if you need to discover table names.
    """
    if not DB_SCHEMA:
        return "Error: schema dictionary not built yet"

    results: list[str] = []
    for table in (t.strip() for t in table_names.split(",")):
        if table not in DB_SCHEMA:
            results.append(f"Error: table {table!r} not found in database")
            continue
        results.append(_format_schema(table, DB_SCHEMA[table]))

    return "\n\n".join(results)


@function_tool
def sql_db_query(query: str) -> str:
    """Execute a **read-only** SQL query against the database.

    Input: a complete, syntactically correct ``SELECT`` statement.
    Returns the result rows as a Python list-of-tuples string, or an error
    message if the query fails.

    **Always** run the query through ``sql_db_query_checker`` before calling
    this tool.
    """
    with _get_db() as con:
        cur = con.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
            return str(rows)
        except Exception as e:
            return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Specialist Sub-Agent: SQL Query Checker
# ═══════════════════════════════════════════════════════════════════════════════

_query_checker_instructions = textwrap.dedent("""\
    You are a meticulous SQL query reviewer.  Your **only** job is to
    double-check a SQL query for common mistakes.

    Check for these issues:
    * ``NOT IN`` with ``NULL`` values (rewrite using ``NOT EXISTS``)
    * ``UNION`` when ``UNION ALL`` should have been used
    * ``BETWEEN`` with exclusive ranges
    * Data-type mismatches in predicates (e.g. comparing text to integer)
    * Improperly quoted identifiers
    * Wrong number of arguments for built-in functions
    * Missing or incorrect ``CAST`` expressions
    * Incorrect join columns (check against the foreign keys in the schema)

    Rules:
    * If you find a mistake, output the **corrected** query.
    * If the query is already correct, output the original query unchanged.
    * **Output only the final SQL query — no markdown, no explanation, no commentary.**
""")

query_checker_agent = Agent(
    name="SQL Query Checker",
    instructions=_query_checker_instructions,
    model=_openrouter_model,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Main SQL Agent
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    """Build the system prompt — no table list, agent discovers via search."""
    n_tables = len(DB_SCHEMA)
    return textwrap.dedent(f"""\
        You are an agent that answers questions about a SQLite database by
        writing and running queries.  Follow this workflow **in order**:

        1. Call ``sql_db_search_tables`` with keywords from the user's question
           to find relevant tables.  (The database has {n_tables} tables —
           don't guess names, always search first.)
        2. Call ``sql_db_schema`` for the tables found in step 1.
        3. Write a SQL query based on the schema.
        4. Call ``sql_db_query_checker`` to review your query for mistakes.
        5. Execute the reviewed query with ``sql_db_query``.
        6. If the query fails, fix it and retry from step 4.
        7. Form a natural-language answer from the results.

        Rules:
        * Never ``SELECT *`` — list only the columns you actually need.
        * Limit results to 5 rows unless the user asks for more.
        * Order results meaningfully (e.g. by the metric the user cares about).
        * **Never** run ``INSERT``, ``UPDATE``, ``DELETE``, ``DROP``, or any DML.
    """)


sql_agent = Agent(
    name="SQL Agent",
    instructions="",  # will be set in main() after schema is built
    tools=[
        sql_db_search_tables,
        sql_db_list_tables,
        sql_db_schema,
        sql_db_query,
        query_checker_agent.as_tool(
            tool_name="sql_db_query_checker",
            tool_description=(
                "Double-check a SQL query for common mistakes before executing it. "
                "Always call this RIGHT BEFORE ``sql_db_query``. "
                "Input: the SQL query to check. Output: the corrected (or original) query."
            ),
        ),
    ],
    model=_openrouter_model,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _describe_tools(agent: Agent) -> str:
    lines: list[str] = []
    for t in agent.tools:
        lines.append(f"  {t.name}")
    return "\n".join(lines)


async def main() -> None:
    download_chinook_db()

    # Load schema catalog from TOML file (single source of truth)
    print("Loading schema catalog ...", end=" ", flush=True)
    global DB_SCHEMA
    DB_SCHEMA = load_schema_catalog(SCHEMA_TOML)
    print(f"done ({len(DB_SCHEMA)} tables)")

    # Inject the table summary into the agent's system prompt
    sql_agent.instructions = _build_system_prompt()

    print()
    print("=" * 64)
    print("  SQL Agent -- OpenAI Agents SDK + OpenRouter")
    print("=" * 64)
    print(f"  Provider  : OpenRouter (api/v1)")
    print(f"  Model     : {MODEL}")
    print(f"  Database  : {DB_PATH}")
    print(f"  Main agent: {sql_agent.name}")
    print(f"  Tools     :")
    print(_describe_tools(sql_agent))
    print(f"    (sql_db_query_checker is a specialist *sub-agent* called as a tool)")
    print("=" * 64)

    questions: Sequence[str] = [
        "Which genre on average has the longest tracks?",
        "How many customers are from each country? Show the top 5.",
    ]

    for idx, question in enumerate(questions, 1):
        print(f"\n-- Query {idx} " + "-" * 50)
        print(f"Q: {question}\n")

        result = await Runner.run(sql_agent, question)

        for item in result.new_items:
            if item.type == "tool_call_item":
                name = item.tool_name or "?"
                raw = item.raw_item
                args = raw.get("arguments", "") if isinstance(raw, dict) else getattr(raw, "arguments", "")
                args_str = str(args)[:120]
                print(f"  [tool] {name}({args_str}{'...' if len(str(args)) > 120 else ''})")
            elif item.type == "tool_call_output_item":
                out = str(item.output)[:200].encode("ascii", errors="replace").decode()
                out_disp = out.replace("\n", " ")
                print(f"      -> {out_disp}{'...' if len(str(item.output)) > 200 else ''}")

        final = result.final_output.encode("ascii", errors="replace").decode()
        print(f"\n  [answer] {final}")
        print()

    print("-" * 64)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())