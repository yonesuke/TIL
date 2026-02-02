# Claude Code Skill作成方法

**Date:** 2026-02-02

## 概要

Claude Codeのスキル機能を使って、独自のコマンド（例: `/til`）を作成する方法についてまとめる。スキルは `~/.claude/skills/` ディレクトリにMarkdownファイルとして配置することで利用できる。

## 詳細

### スキルとは

Claude Codeのスキルは、特定のタスクを自動化するためのカスタムコマンド。ユーザーが `/skill-name` と入力すると、対応する `SKILL.md` の指示に従ってClaudeが動作する。

### ファイル構成

```
~/.claude/skills/
└── my-skill/
    ├── SKILL.md           # メインの指示ファイル（必須）
    ├── templates/         # テンプレートファイル（任意）
    │   ├── template.md
    │   └── template.ipynb
    └── scripts/           # 補助スクリプト（任意）
```

### SKILL.mdの構造

```markdown
---
name: skill-name
description: スキルの説明文
disable-model-invocation: false  # trueでユーザーのみ呼び出し可能
user-invocable: true            # falseでClaudeのみ呼び出し可能
---

# スキル名

## Usage

使い方の説明

## Instructions

Claudeへの具体的な指示
```

### Frontmatterオプション

| オプション | 説明 |
|-----------|------|
| `name` | スキルの識別子 |
| `description` | スキルの説明 |
| `disable-model-invocation` | `true`でユーザーのみ呼び出し可能（副作用のある操作向け） |
| `user-invocable` | `false`でClaudeのみ呼び出し可能（背景知識として利用） |
| `allowed-tools` | 使用可能なツールを制限 |
| `context: fork` | 独立したサブエージェントで実行 |

### 呼び出しパターン

1. **デフォルト（両方可能）**: 一般的なスキル
2. **ユーザーのみ** (`disable-model-invocation: true`): デプロイ、git push など副作用のある操作
3. **Claudeのみ** (`user-invocable: false`): プロジェクトの規約など背景知識

### 実例: TILスキル

```markdown
---
name: til
description: TILリポジトリに新しいエントリを追加
---

# TIL Skill

## Usage

/til "トピック名"
/til  # 会話から自動抽出

## Instructions

1. トピックを特定（引数または会話履歴から）
2. リポジトリをクローン
3. 日付フォルダを作成（例: 20260202_topic_name）
4. テンプレートからファイル生成
5. git commit & push
6. クリーンアップ
```

### ベストプラクティス

- **明確な指示**: Claudeが迷わないよう具体的に記述
- **エラーハンドリング**: 失敗時の対処法も記載
- **テンプレート活用**: 一貫したフォーマットのためにテンプレートを用意
- **副作用の明示**: git pushなど元に戻せない操作は明確に

## 参考資料

- [Claude Code公式ドキュメント](https://docs.anthropic.com/en/docs/claude-code)
- スキルリファレンス: `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/claude-code-setup/skills/claude-automation-recommender/references/skills-reference.md`

## メモ

- スキルは `~/.claude/skills/` に配置するだけで自動的に認識される
- 会話のコンテキストを活用することで、より賢いスキルが作れる
- 複数のスキルを組み合わせて複雑なワークフローも構築可能
