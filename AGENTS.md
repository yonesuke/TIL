# AGENTS.md

このドキュメントはAIエージェント（Claude Code等）がこのリポジトリで作業する際のガイドラインです。

## リポジトリ概要

TIL（Today I Learned）リポジトリは、日々学んだことを記録するためのリポジトリです。

## ディレクトリ構成

各トピックは `YYYYMMDD_topic` 形式のディレクトリに配置します。

```
TIL/
├── YYYYMMDD_topic/
│   ├── YYYYMMDD_topic.md    # 説明・ノート
│   └── YYYYMMDD_topic.py    # 実装コード（Pythonの場合）
├── README.md
└── AGENTS.md
```

### 命名規則

- **ディレクトリ名**: `YYYYMMDD_トピック名`（例: `20260126_tsmixer`）
- **ファイル名**: ディレクトリ名と同じベース名を使用
  - Markdown: `YYYYMMDD_topic.md`
  - Python: `YYYYMMDD_topic.py`
  - その他: `YYYYMMDD_topic.ext`

## Pythonファイルの書き方

### PEP 723 (Inline Script Metadata)

Pythonファイルは**PEP 723**形式を使用して、シングルファイルで実行可能にします。これにより`uv run`コマンドで依存関係の自動インストールと実行が可能になります。

### フォーマット

ファイルの先頭に以下の形式でメタデータを記述します：

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax>=0.4.20",
#     "flax>=0.10.0",
#     "optax",
#     "numpy",
# ]
# ///
"""
モジュールのドキュメント文字列

実装の概要や参考文献をここに記述します。
"""

import jax
# ... 以降のコード
```

### 実行方法

```bash
# uv runで直接実行（依存関係は自動でインストールされる）
uv run 20260126_tsmixer/20260126_tsmixer.py
```

### PEP 723のポイント

1. `# /// script` と `# ///` でメタデータブロックを囲む
2. `requires-python` でPythonバージョンを指定
3. `dependencies` で必要なパッケージをリスト形式で記述
4. バージョン指定は必要に応じて `>=`, `==`, `~=` 等を使用

## 新しいエントリの追加手順

1. **ディレクトリ作成**: `YYYYMMDD_topic` 形式で新しいディレクトリを作成
2. **ファイル作成**:
   - `YYYYMMDD_topic.md`: トピックの説明、理論的背景、参考文献
   - `YYYYMMDD_topic.py`: 実装コード（PEP 723形式）
3. **README更新**: `README.md` の目次とカテゴリ別セクションに新しいエントリを追加

## READMEの更新

新しいエントリを追加する際は、`README.md`の以下のセクションを更新してください：

### 目次セクション

日付順にエントリを追加：

```markdown
| 日付 | トピック | 内容 |
|------|----------|------|
| YYYY-MM-DD | [トピック名](./YYYYMMDD_topic/) | 簡潔な説明 |
```

### カテゴリ別セクション

適切なカテゴリにエントリを追加：

- **物理学・力学系**: 統計力学、熱力学、固体物理など
- **微分方程式・力学系**: ODE、非線形ダイナミクスなど
- **数学**: 解析学、代数、幾何など
- **機械学習・計算**: ML、強化学習、数値計算など
- **その他**: 上記に当てはまらないもの

## コードスタイル

- 型ヒントを使用する
- docstringを記述する
- `if __name__ == "__main__":` でエントリポイントを定義
- 実行可能なデモ/例を含める

## 推奨ライブラリ

機械学習関連の実装では以下を優先的に使用：

- **JAX**: 数値計算、自動微分
- **Flax NNX**: ニューラルネットワーク
- **Optax**: 最適化
- **NumPy**: 基本的な配列操作

---

## 汎用的な知識・Tips

### Matplotlib日本語フォント設定

システムに日本語フォントがインストールされていない環境でも、Google FontsからNoto Sans JPをダウンロードして使用できる。

```python
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import requests
from pathlib import Path

def setup_japanese_font() -> None:
    """Google FontsからNoto Sans JPをダウンロードしてmatplotlibに設定"""
    cache_dir = Path.home() / ".cache" / "matplotlib_jp_fonts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    font_path = cache_dir / "NotoSansJP-Regular.ttf"

    if not font_path.exists():
        url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        font_path.write_bytes(response.content)

    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止
```

**ポイント:**
- `~/.cache/matplotlib_jp_fonts/` にキャッシュして再ダウンロードを防ぐ
- `fm.fontManager.addfont()` で動的にフォントを登録
- `axes.unicode_minus = False` でマイナス記号の表示問題を回避

---

## 金融データ分析

### TOPIX-17セクターETF

野村アセットのTOPIX-17業種別ETF一覧（yfinanceで取得可能）:

| ティッカー | セクター |
|-----------|---------|
| 1617.T | 食品 |
| 1618.T | エネルギー資源 |
| 1619.T | 建設・資材 |
| 1620.T | 素材・化学 |
| 1621.T | 医薬品 |
| 1622.T | 自動車・輸送機 |
| 1623.T | 鉄鋼・非鉄 |
| 1624.T | 機械 |
| 1625.T | 電機・精密 |
| 1626.T | 情報通信・サービス他 |
| 1627.T | 電力・ガス |
| 1628.T | 運輸・物流 |
| 1629.T | 商社・卸売 |
| 1630.T | 小売 |
| 1631.T | 銀行 |
| 1632.T | 金融（除く銀行） |
| 1633.T | 不動産 |

### リターン計算方法

| 方法 | 計算式 | 用途 |
|-----|--------|------|
| Close-to-Close | (Close_t - Close_{t-1}) / Close_{t-1} | 標準的なリターン計算 |
| Open-to-Close | (Close_t - Open_t) / Open_t | 日中リターン |

### ランキング回転率（Turnover）指標

ランキングの安定性・持続性を測る指標:

| 指標 | 説明 | 解釈 |
|-----|------|------|
| **Spearman順位相関** | 前期と今期のランキング間の相関 | 1=完全一致、0=無相関、-1=完全逆転 |
| **Kendall's tau** | 順位ペアの一致度 | 同上 |
| **Top-N回転率** | 上位N位の入れ替わり率 | 0=変化なし、1=全入替 |
| **平均順位変化** | 全銘柄の平均的な順位変動幅 | 大きいほど不安定 |
| **順位変化RMS** | 順位変化の二乗平均平方根 | 大きな変動に敏感 |

```python
from scipy import stats

# Spearman順位相関
corr, _ = stats.spearmanr(prev_rank, curr_rank)

# Kendall's tau
tau, _ = stats.kendalltau(prev_rank, curr_rank)

# Top-N回転率
prev_top_n = set(prev_rank[prev_rank <= n].index)
curr_top_n = set(curr_rank[curr_rank <= n].index)
turnover = 1 - len(prev_top_n & curr_top_n) / n
```

### yfinanceでのデータ取得

```python
import yfinance as yf

ticker = yf.Ticker("1617.T")
df = ticker.history(period="max")  # 全期間取得
# columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
```
