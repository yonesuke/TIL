# 基底関数線形回帰と単体制約付き二次計画法（QP）によるヨーロピアン・オプションの最適ヘッジ

## 概要

本稿では、ヨーロピアン・コールオプションの動的デルタヘッジ戦略を、**状態変数 $(S, t)$ の基底関数の線形結合**として定式化し、**モンテカルロシミュレーションによるヘッジ損益（PnL）の平均分散ポートフォリオ最適化（二次計画問題: Quadratic Programming）**として解く手法についてまとめます。

さらに、Black-Scholesの理論式を知らない（モデルフリーな）状況下で、**「単調性（$\frac{\partial \Delta}{\partial S} \ge 0$）」**および**「漸近挙動（$S \to 0$ で $0$、$S \to \infty$ で $1$）」**を厳密に保証する**汎用ロジスティック・シグモイド基底と単体制約（Simplex Constraint）**の設計について解説します。

---

## 1. 問題設定とヘッジ PnL の線形表現

### 1.1 設定
- 原資産価格過程: $S_t$ （幾何ブラウン運動または任意のシミュレーション可能過程）
- ヘッジ期間: $t_0 = 0 < t_1 < \dots < t_N = T$ （等間隔 $\Delta t = T / N$）
- 割引係数: $D_k = e^{-r t_k}$、割引株価: $\tilde{S}_k = D_k S_k$
- 満期ペイオフ（割引）: $y = e^{-rT} (S_T - K)^+$

### 1.2 ヘッジ比率の基底展開
各ステップ $t_k$ での株の保有量 $\Delta(S_k, t_k)$ を $D$ 次元の基底関数 $\boldsymbol{\phi}(S, t)$ の線形結合としてモデル化します：
$$\Delta(S_k, t_k) = \sum_{j=1}^D w_j \phi_j(S_k, t_k) = \mathbf{w}^\top \boldsymbol{\phi}(S_k, t_k)$$

### 1.3 満期割引 PnL
初期受取プレミアム（オプション価格）を $c_0$ とすると、オプションショート側の満期割引ヘッジ PnL $\Pi$ は：
$$\Pi = c_0 + \sum_{k=0}^{N-1} \Delta(S_k, t_k) (\tilde{S}_{k+1} - \tilde{S}_k) - y = c_0 + \sum_{j=1}^D w_j \underbrace{\left( \sum_{k=0}^{N-1} \phi_j(S_k, t_k) \Delta \tilde{S}_k \right)}_{X_j} - y$$

ここで、$X_j$ は**「$j$ 番目の基底関数戦略を単体で運用したときの累積割引ゲイン」**を表します。
モンテカルロ法で $M$ 本のパスを生成すると、パス $m$ ごとの累積ゲインベクトル $\mathbf{X}_m \in \mathbb{R}^D$ とペイオフ $y_m$ が得られ、PnL は完全に線形モデルとして書けます：
$$\Pi_m = c_0 + \mathbf{X}_m^\top \mathbf{w} - y_m$$

---

## 2. 平均分散ポートフォリオ（二次計画問題: QP）への定式化

### 2.1 期待値と分散の二次形式展開
各統計量を $\bar{\mathbf{X}} = \mathbb{E}[\mathbf{X}]$, $\bar{y} = \mathbb{E}[y]$, $\boldsymbol{\Sigma}_{XX} = \operatorname{Cov}(\mathbf{X}, \mathbf{X})$, $\boldsymbol{\Sigma}_{Xy} = \operatorname{Cov}(\mathbf{X}, y)$, $\sigma_y^2 = \operatorname{Var}(y)$ とおくと：

$$\mathbb{E}[\Pi] = c_0 - \bar{y} + \bar{\mathbf{X}}^\top \mathbf{w}$$
$$\operatorname{Var}(\Pi) = \mathbf{w}^\top \boldsymbol{\Sigma}_{XX} \mathbf{w} - 2 \boldsymbol{\Sigma}_{Xy}^\top \mathbf{w} + \sigma_y^2$$

### 2.2 平均分散目的関数の変形
リスク回避度 $\lambda > 0$ に対する平均分散目的関数：
$$\max_{\mathbf{w}} \left( \mathbb{E}[\Pi] - \frac{\lambda}{2} \operatorname{Var}(\Pi) \right) \quad \Longleftrightarrow \quad \min_{\mathbf{w}} \left( \frac{1}{2} \mathbf{w}^\top \mathbf{Q} \mathbf{w} + \mathbf{q}^\top \mathbf{w} \right)$$

ここで：
$$\mathbf{Q} = \lambda \boldsymbol{\Sigma}_{XX}, \qquad \mathbf{q} = - \left( \bar{\mathbf{X}} + \lambda \boldsymbol{\Sigma}_{Xy} \right)$$

### 2.3 制約なし解の分解
$$\mathbf{w}^*(\lambda) = \underbrace{\boldsymbol{\Sigma}_{XX}^{-1} \boldsymbol{\Sigma}_{Xy}}_{\text{純粋ヘッジ項 (最小分散 / 線形回帰係数)}} + \frac{1}{\lambda} \underbrace{\boldsymbol{\Sigma}_{XX}^{-1} \bar{\mathbf{X}}_{\vphantom{y}}}_{\text{スペキュレーション項 (ドリフト獲得)}}$$

- $\lambda \to \infty$（純粋ヘッジ）: 満期ペイオフとの共分散を相殺する最小分散ヘッジ（OLS回帰係数）に一致。
- 有限の $\lambda$: 原資産のドリフト $\mu > r$（$\bar{\mathbf{X}} > 0$）を取りに行くロングポジションが上乗せされる。

---

## 3. 基底エンジニアリングと単体制約（Simplex Constraint）

### 3.1 多項式基底の限界
単純な多項式基底（$S, S^2, t, St, \dots$）では：
1. 深外・深内で $\pm \infty$ に発散する（有界性 $[0, 1]$ の欠如）
2. 満期直前（$\tau \to 0$）のステップ関数形状（急激なガンマの立ち上がり）を滑らかに近似できず波打つ（ギブズ現象）

### 3.2 汎用ロジスティック・シグモイド基底
特定の確率分布（正規分布）を仮定せず、機械学習の標準的なシグモイド関数 $\sigma(z) = \frac{1}{1 + e^{-z}}$ を空間・時間に配置します：
$$\phi_{k, m}(S, t) = \sigma\left( \frac{S - s_k}{\delta_m \cdot K \sqrt{(T-t)/T}} \right)$$
- 空間中心 $s_k$: ATM（$K$）周辺を高密度にサンプリング（$0.7K \sim 1.3K$）
- 幅スケール $\delta_m$: 局所的な急峻ステップから大域的な滑らかさまでマルチスケール配置
- 時間スケーリング: 拡散の標準偏差スケール $\sqrt{\tau}$ で正規化

### 3.3 単体制約（Simplex Constraint）による物理的完全保証
基底 $\phi_j$ は単体で「$0 \to 1$ の単調増加関数」であるため、重み $\mathbf{w}$ に**単体制約（凸結合制約）**を課して QP を解きます：
$$\min_{\mathbf{w}} \left( \frac{1}{2} \mathbf{w}^\top \mathbf{Q} \mathbf{w} + \mathbf{q}^\top \mathbf{w} \right) \quad \text{subject to} \quad w_j \ge 0, \quad \sum_{j=1}^D w_j = 1$$

#### 数学的メリット：
1. **$0 \le \Delta(S, t) \le 1$ の完全保証**: デルタが $[0, 1]$ を逸脱することが数学的にあり得ない。
2. **単調増加性（$\frac{\partial \Delta}{\partial S} \ge 0$）の完全保証**: 単調増加関数の非負結合は必ず単調増加。
3. **自動スパース選択（$L_1$ 正則化効果）**: 多数の基底候補（辞書）を用意しても過学習せず、ヘッジに本当に必要な数個の基底のみが自動選択される。

---

## 4. 実装と実験結果

### 4.1 実験設定
- 原資産パラメータ: $S_0 = 100, K = 100, T = 1.0\text{年}, r = 5\%, \mu = 10\%, \sigma = 20\%$
- モンテカルロ: $M = 100,000$ パス, $N = 250$ ステップ（日次ヘッジ）
- シグモイド基底次元: $D = 160$（ATM近傍16中心 $\times$ 5スケール $\times$ 2時間変調）

### 4.2 結果サマリー（日次ヘッジ $N=250$）

| ヘッジ手法 | モデル知識 | PnL 標準偏差 | **1% PnL Quantile ($Q_{0.01}$)** | BS理論値との 1%Q 差 |
| :--- | :---: | :---: | :---: | :---: |
| **Black-Scholes 理論デルタ（正解）** | 完全（理論式） | **0.4065** | **-1.0925** | — |
| **従来の多項式基底 ($D=10$)** | なし（モデルフリー） | 1.5598 | -2.3766 | 1.2841 |
| **粗い汎用シグモイド ($D=45$)** | なし（モデルフリー） | 0.7602 | -1.9689 | 0.8765 |
| **高次元汎用シグモイド ($D=160$)** | なし（モデルフリー） | **0.4691** | **-1.1791** | **0.0866** |

- 高次元汎用シグモイド（$D=160$）では、160個の候補から **わずか 9 個の有効基底** が単体制約によって自動スパース選択され、**BS理論解とのテールリスク差がわずか 0.0866（ほぼ理論限界）** に到達しました。

---

## 5. 実行方法

PEP 723 形式に対応しているため、`uv run` で直接実行可能です：

```bash
uv run 20260818_mean_variance_hedging.py
```
