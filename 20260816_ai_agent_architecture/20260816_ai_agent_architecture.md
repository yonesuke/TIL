---
marp: true
theme: gaia
_class: lead
paginate: true
size: 16:9
backgroundColor: #0f172a
color: #f8fafc
style: |
  section {
    font-family: 'Inter', 'Noto Sans JP', sans-serif;
    padding: 44px 60px;
    background-color: #0f172a;
    color: #f8fafc;
    font-size: 27px;
    line-height: 1.5;
  }
  h1 {
    color: #38bdf8;
    font-size: 1.9em;
    font-weight: 800;
    margin-bottom: 0.3em;
    line-height: 1.2;
  }
  h2 {
    color: #818cf8;
    font-size: 1.4em;
    font-weight: 700;
    border-bottom: 2px solid #334155;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 0.5em;
  }
  h3 {
    color: #38bdf8;
    font-size: 1.1em;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 0.35em;
  }
  p, li {
    font-size: 0.92em;
    line-height: 1.5;
  }
  ul, ol {
    margin-top: 0.2em;
    margin-bottom: 0.4em;
    padding-left: 1.2em;
  }
  pre {
    background-color: #1e293b !important;
    border: 1px solid #334155;
    border-radius: 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
    font-size: 0.62em !important;
    line-height: 1.45 !important;
    padding: 16px 20px !important;
    margin: 0.3em 0 !important;
  }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
    color: #38bdf8;
    font-size: 0.9em;
  }
  table {
    font-size: 0.82em;
    width: 100%;
    margin-top: 0.5em;
    background-color: #1e293b;
    border-collapse: collapse;
  }
  th {
    background-color: #334155;
    color: #38bdf8;
    padding: 10px 14px;
    font-weight: 700;
  }
  td {
    border: 1px solid #334155;
    padding: 10px 14px;
  }
  footer {
    font-size: 0.45em;
    color: #64748b;
    bottom: 15px;
  }
  .highlight {
    background-color: rgba(56, 189, 248, 0.12);
    border-left: 4px solid #38bdf8;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.88em;
    margin-top: 14px;
  }
  .highlight-amber {
    background-color: rgba(251, 191, 36, 0.12);
    border-left: 4px solid #fbbf24;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.88em;
    margin-top: 14px;
    color: #fef08a;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
  }
  .card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px 20px;
  }
  .mermaid {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px;
    display: flex;
    justify-content: center;
    font-size: 0.85em;
  }
  .mermaid svg {
    max-width: 100% !important;
    max-height: 420px !important;
  }
header: 'AI Agent Architecture for Engineers'
footer: '© 2026 Tech Seminar | AI Agent Deep Dive'
---

<!-- _class: lead -->
<!-- _paginate: false -->

# コードで読み解く<br>AIエージェントのアーキテクチャ
### 〜2つのWhileループからコーディングエージェントの実装・企業応用まで〜

<div style="margin-top: 20px; font-size: 0.85em; color: #94a3b8;">
Engineering Seminar 2026
</div>

---

## 本日のアジェンダ

<div class="grid">
<div class="card">

### 1. エージェントとは何か？
- ChatBot vs RPA vs AIエージェント
- 古典的定義とReAct論文のブレイクスルー

### 2. 内部構造とプロトコル
- 2重Whileループの最小実装コード
- Tool CallingプロトコルとMCP規格
</div>

<div class="card">

### 3. 最前線：コーディングエージェント
- 成功の理由と基本ツールセット
- 差分置換とコンテキスト管理の妙味

### 4. 企業開発・本番運用のリアル
- サンドボックス隔離と権限管理
- 社内SRE/データ分析への応用
</div>
</div>

---

## 1.1 そもそも「エージェント」とは何か？

<div class="grid">
<div class="card">

### 古典的AIの定義 (Russell & Norvig)
> **「環境（Environment）を観測（Perceive）し、判断（Reason）し、行動（Act）を起こす主体」**

- LLM単体は「テキスト予測器」に過ぎない
- **ツール（Act）**と**実行結果（Perceive）**を与え、ループさせることで初めてエージェントになる
</div>

<div class="card">

### なぜ「今」実用化したのか？
- **推論能力の向上**: 指示追従（Instruction Following）の成熟
- **Tool Callingの標準化**: JSON Schemaによる確定的な引数生成
- **ReAct論文 (2022)**: 「思考 $\rightarrow$ 行動 $\rightarrow$ 観察」の反復がハルシネーションを激減
</div>
</div>

<div class="highlight">
💡 <strong>Point:</strong> 「おしゃべりするAI（Chat）」から「環境の状態を変更するAI（Agent）」への進化
</div>

---

## 1.2 ChatBot vs RPA vs AIエージェント

| 項目 | ChatBot (従来型LLM) | RPA / ルールベース | AIエージェント |
| :--- | :--- | :--- | :--- |
| **入力/出力** | テキスト $\rightarrow$ テキスト | トリガー $\rightarrow$ 固定手順 | **目標（Goal） $\rightarrow$ 達成状態** |
| **実行形態** | 1ターンの推論（一問一答） | 静的な分岐（if-else） | **自律ループ（計画 $\rightarrow$ 実行 $\rightarrow$ 修正）** |
| **状態管理** | ステートレス（会話履歴） | 固定変数 | **環境の変化（State）を追跡** |
| **エラー対応** | 人間が聞き直す | 即座に例外停止 | **エラーログを読み自己修復** |

<div class="highlight">
🔑 <strong>最大の違い:</strong> エラーが起きた時、スタックトレースを読んで「自律的に別のアプローチを試せるか」
</div>

---

## 2.1 エージェントの最小実装（2つのWhileループ）

エージェントの本質は、驚くほどシンプルな**「2重Whileループ」**。

<!-- _style: "pre { font-size: 0.60em !important; line-height: 1.4 !important; padding: 14px 18px !important; margin: 0 !important; }" -->
```python
messages = [{"role": "system", "content": "You are a helpful assistant."}]

# 【第1のループ】：ユーザーとの対話を受け付ける外側ループ
while True:
    user_input = input("\nUser > ")
    messages.append({"role": "user", "content": user_input})

    # 【第2のループ】：ツール実行と推論を繰り返す内側ループ（エージェントループ）
    while True:
        response = client.chat.completions.create(
            model="gpt-4o / claude-3-7-sonnet",
            messages=messages,
            tools=TOOLS_DEFINITION  # JSON Schemaで定義した関数群
        )
        msg = response.choices[0].message
        messages.append(msg)

        # ツール呼び出しがなければ完了 $\rightarrow$ ループを抜けて回答出力
        if not msg.tool_calls:
            print(f"\nAgent > {msg.content}")
            break

        # ツール呼び出しがあればローカルで実行し、結果を履歴に追加して再推論
        for tool_call in msg.tool_calls:
            output = execute_local_function(tool_call.function.name, tool_call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(output)})
```

---

## 2.2 ツール実行プロトコルのシーケンス

<pre class="mermaid">
sequenceDiagram
    autonumber
    actor User as ユーザー
    participant Runner as アプリ (Runner)
    participant LLM as LLM (Brain)
    participant Tool as ツール (OS/DB/API)

    User->>Runner: "売上上位3社を出して"
    Runner->>LLM: messages + tools (Schema)
    LLM-->>Runner: tool_calls: query_db(sql="SELECT...")
    Runner->>Tool: ローカルでSQL実行
    Tool-->>Runner: 結果: [{"name":"A社","sales":100}]
    Runner->>LLM: messages + role: 'tool' (結果)
    LLM-->>Runner: "上位3社はA社(100万)..." (完了)
    Runner-->>User: 最終回答を表示
</pre>

<div class="highlight">
⚠️ <strong>重要:</strong> LLMが直接DBやAPIを叩くのではない。LLMは「引数付きのJSON」を返すだけで、実行するのは手元のRunner。
</div>

---

## 2.3 ワークフロー設計パターン (Anthropic提唱)

すべてを「自由気ままな自律エージェント」にする必要はありません。

<div class="grid-3">
<div class="card">

### ① Chaining & Routing
前工程の出力を次へ渡す直列処理や、入力意図に応じて特化プロンプト/ツールへ分岐。
</div>

<div class="card">

### ② Orchestrator-Workers
親エージェントがタスクを分解し、複数のWorkerに並列委任して結果を集約。
</div>

<div class="card">

### ③ Evaluator-Optimizer
生成役（Generator）と採点役（Evaluator）を分け、基準を満たすまでループ。
</div>
</div>

<div class="highlight-amber">
⚠️ <strong>実務の鉄則:</strong> 8割の確定的なコードフロー ＋ 2割の局所的エージェントループが最も安定する
</div>

---

## 3.1 なぜコーディングでエージェントが先行成功したのか？

<div class="grid">
<div class="card">

### 成功の3大要因
1. **完全にデジタルな環境**
   - ファイル、Git、CLIがすべてコマンドで操作可能
2. **客観的・確定的なフィードバック**
   - `SyntaxError: line 42`
   - `Tests: 1 failed, 4 passed`
3. **自己修復が自然に回る**
   - スタックトレースを渡すだけでAIが次の試行で修正
</div>

<div class="card">

### エージェントに渡す基本ツール
```bash
view_file(path, start, end)
# 行番号付きで安全に部分読み込み

replace_file_content(path, diff)
# ピンポイント差分置換

write_to_file(path, content)
# 新規ファイル作成

grep_search(query, path)
# ripgrepで高速全文検索

run_command(cmd, cwd)
# bashでビルド・テスト実行
```
</div>
</div>

---

## 3.2 ツール設計のエンジニアリングの妙味

<div class="grid">
<div class="card">

### ① 「丸ごと上書き」を避ける差分編集
- 数千行のファイルを丸ごと再生成させるとトークン代高騰＆省略ハルシネーションが発生
- $\rightarrow$ **一意な文字列置換（TargetContent $\rightarrow$ ReplacementContent）** が必須
</div>

<div class="card">

### ② コンテキストウィンドウ枯渇対策
- 長時間タスクでは128k/200kトークンがすぐ溢れる
- **Tool Truncation**: 長すぎるログを間引く
- **Subagent**: 調査専用の子エージェントに調べさせ、サマリーだけ受け取る
</div>
</div>

<div class="highlight">
💡 <strong>Point:</strong> プロンプトの文言以上に「Tool設計」と「エラーフィードバックの質」が精度を決定づける
</div>

---

## 4.1 企業導入における「3大セキュリティ・安全対策」

<div class="grid-3">
<div class="card">

### ① サンドボックス隔離
- `run_command` や SQL実行をホストで直接動かさない
- 一時Dockerコンテナ、gVisor、ReadOnly DBレプリカで実行
</div>

<div class="card">

### ② 権限分離 & 人間承認
- 読み取り（Read/Search）は全自動
- 破壊的変更（DB更新、Git Push、決済、メール送信）は**Human-in-the-loop（承認待ち）**
</div>

<div class="card">

### ③ プロンプトインジェクション
- ツール経由で取得した外部Webや社内文書内の悪意ある指示を防御
- ツール戻り値を厳密にデータ領域としてエスケープ
</div>
</div>

---

## 4.2 社内システムへの応用アーキテクチャ例

<div class="grid">
<div class="card">

### SRE / 障害一次調査エージェント
1. **[PagerDuty アラート検知]**
2. Tool: Datadog メトリクス取得
3. Tool: k8s Pod ログ取得
4. Tool: GitHub 直近コミット差分確認
5. **[Slackに原因候補と推奨対応を即時投稿]**
</div>

<div class="card">

### 社内データ分析・SQL自動生成
1. **[ユーザーの自然言語質問]**
2. Tool: スキーマ検索 $\rightarrow$ SQL生成
3. Tool: ReadOnly DBで実行
4. *(Syntax Error時は自己修正して再試行)*
5. **[集計テーブル ＋ グラフを自動描画]**
</div>
</div>

---

<!-- _class: default -->
## 5. まとめ：エンジニアとしての持ち帰りメッセージ

<div class="grid-3">

<div class="card">

### ① 2つのWhileループ
**エージェントの正体は明確な制御構造。**
過度な幻想も恐怖も不要。中身は完全に制御可能なソフトウェア工学。
</div>

<div class="card">

### ② Tool & フィードバック
**精度を決めるのは「ツール設計」。**
適切な粒度のツールと、客観的エラーフィードバックの設計が成否を分ける。
</div>

<div class="card">

### ③ 段階的ハイブリッド
**決定的なワークフローから始める。**
小さく安全なループから組み込み、人の承認を挟みながら段階的に拡張する。
</div>

</div>

<div class="highlight" style="margin-top: 20px;">
🚀 <strong>Next Action:</strong> 社内の身近なルーチン業務（障害調査・SQL分析等）で小さなループを試作してみよう！
</div>

<!-- Marp公式推奨のMermaid ESM埋め込みスクリプト -->
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({
    startOnLoad: true,
    theme: 'dark',
    themeVariables: {
      darkMode: true,
      background: '#1e293b',
      primaryColor: '#1e293b',
      primaryBorderColor: '#38bdf8',
      primaryTextColor: '#f8fafc',
      lineColor: '#38bdf8',
      actorBkg: '#1e293b',
      actorBorder: '#38bdf8',
      actorTextColor: '#f8fafc',
      signalColor: '#38bdf8',
      signalTextColor: '#f8fafc',
      labelBoxBkgColor: '#1e293b',
      labelBoxBorderColor: '#38bdf8',
      labelTextColor: '#f8fafc',
      noteBkgColor: '#0f172a',
      noteBorderColor: '#818cf8',
      noteTextColor: '#f8fafc'
    }
  });
</script>
