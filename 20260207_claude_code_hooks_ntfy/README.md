# Claude Code の Hooks で ntfy.sh 通知

**Date:** 2026-02-07

## 概要

Claude Code の Hooks 機能を使って、タスク完了時に ntfy.sh へプッシュ通知を送る仕組みを構築した。長時間の処理が終わったときにスマホで通知を受け取れるようになる。

## 詳細

### キーポイント

- Claude Code には `hooks` 機能があり、特定のイベント（`Stop` など）発生時にシェルスクリプトを実行できる
- 設定は `~/.claude/settings.json` の `hooks` フィールドに記述する
- `Stop` イベントは Claude の応答が完了した時に発火する
- hook スクリプトには stdin から JSON が渡され、`transcript_path` でセッションの会話ログにアクセスできる
- [ntfy.sh](https://ntfy.sh) は無料のプッシュ通知サービスで、curl で簡単に通知を送れる

### 設定例

`~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/scripts/notify-ntfy.sh"
          }
        ]
      }
    ]
  }
}
```

### スクリプトの要点

```bash
# stdin から hook の入力を読み取る
INPUT=$(cat)

# transcript のパスを取得
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')

# JSONL 形式の transcript から最後のアシスタントメッセージを取得
MESSAGE=$(tail -r "$TRANSCRIPT_PATH" \
  | grep '"type":"assistant"' \
  | while read -r line; do
      text=$(echo "$line" | jq -r '
        [.message.content[] | select(.type == "text") | .text] | join("\n")
      ' 2>/dev/null)
      if [ -n "$text" ] && [ "$text" != "null" ]; then
        echo "$text"
        break
      fi
    done)

# ntfy.sh に送信
curl -s -H "Title: Claude Code" -d "$MESSAGE" "ntfy.sh/${NTFY_TOPIC}"
```

## 参考資料

- [ntfy.sh](https://ntfy.sh) - プッシュ通知サービス
- [Claude Code Hooks ドキュメント](https://docs.anthropic.com/en/docs/claude-code/hooks)

## メモ

- `stop_hook_active` フラグで無限ループを防止する必要がある
- macOS には `tac` がないので `tail -r` を使う
- デバッグ時はログファイルに入力を書き出すと便利
- transcript は JSONL 形式で、各行が1つのメッセージに対応している
