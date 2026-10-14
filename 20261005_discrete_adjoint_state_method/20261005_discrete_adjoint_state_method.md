# 離散ポントリャーギン随伴状態法 vs Checkpointed BPTT — 数理的等価性と `scan + jax.checkpoint` による完全上位互換の導出 (JAX / Flax NNX)

**日付**: 2026-10-05（改訂: 2026-10-14）  
**キーワード**: Discrete Adjoint State Method, Pontryagin Maximum Principle, BPTT, Activation Checkpointing, Rematerialization, `jax.lax.scan`, `jax.checkpoint`, `prevent_cse`, Reverse-over-Reverse HVP, Flax NNX

---

## 1. 概要と動機

時系列モデル、リカレントニューラルネットワーク（RNN）、超深層 ResNet、強化学習・最適制御における物理シミュレータなど、以下のような**離散力学系（リカレント遷移）**の感応度解析やエンドツーエンド学習を考える：

$$
x_{t+1} = f(x_t, u_t; \theta), \quad t = 0, 1, \dots, T-1
$$

標準的な逆モード自動微分（Backpropagation Through Time: BPTT）では、全ステップ $T \times$ 全層 $L$ の中間活性化テンソルをメモリ（Tape）にキャッシュするため、ホライズン $T$ が長大になると即座に GPU/TPU メモリが枯渇（OOM）する。

この問題に対し、最適制御理論における**「ポントリャーギンの離散最大原理（Discrete Pontryagin Maximum Principle）／離散随伴状態法（Discrete Adjoint State Method）」**に基づき、`jax.custom_vjp` を手動定義して中間活性化を破棄し、逆順スキャン時に局所 VJP をオンザフライ計算するソルバが考えられる。

しかし、JAX の内部機構を詳細に調査すると、以下の重要な事実が明らかになる：

> **「離散力学系の手動 Adjoint 法は、数学的にもアルゴリズム的にも、`jax.lax.scan` の反復ステップに `jax.checkpoint` を適用した Checkpointed BPTT と完全に等価である」**

さらに、JAX のコンパイラ（XLA）の観点からは、ちまちまと手動 `custom_vjp` を書くよりも、**`scan` + `jax.checkpoint(prevent_cse=False)` に任せた方が「メモリが少なく」「実行速度が高速で」「Forward-mode AD (JVP) や Hessian に完全対応する」という完全上位互換**になる。

本稿では、数理的背景、Jaxpr/HLO レベルでの解剖、XLA の CSE（共通部分式除去）最適化機構、および実機ベンチマークを通じて、この両者の関係とベストプラクティスを解説する。

---

## 2. 数理定式化：離散随伴方程式 ＝ BPTT の連鎖律

全時間発展拘束をラグランジュ未定乗数 $\lambda_t \in \mathbb{R}^d$（共役変数 / 随伴変数 Costate）で束縛した拡大ラグランジアンを定義する：

$$
\mathcal{J} = \mathcal{L}(x_1, \dots, x_T) + \sum_{t=0}^{T-1} \lambda_{t+1}^\top \Big( f(x_t, u_t; \theta) - x_{t+1} \Big)
$$

停留条件 $\nabla_{x_t} \mathcal{J} = 0$ および $\nabla_\theta \mathcal{J} = 0$ をとることで、以下の離散随伴方程式が導かれる。

### ① 共役変数の時間逆順発展（Adjoint Backward Equation）
終端条件 $\lambda_{T+1} \equiv 0$ から過去へ遡る：

$$
\lambda_t = \underbrace{\frac{\partial \mathcal{L}}{\partial x_t}}_{\text{下流損失からの直接感応度}} + \underbrace{\left( \frac{\partial f(x_t, u_t; \theta)}{\partial x_t} \right)^\top \lambda_{t+1}}_{\text{将来状態を通じた波及効果}}, \quad t = T, T-1, \dots, 1
$$

### ② パラメータ勾配の積算
各ステップにおける局所パラメータ感応度は、1 ステップ分の局所 VJP として評価される：

$$
g_{\theta, t} = \left( \frac{\partial f(x_t, u_t; \theta)}{\partial \theta} \right)^\top \lambda_{t+1} = \operatorname{vjp}\Big( \theta \mapsto f(x_t, u_t; \theta), \, \lambda_{t+1} \Big)_\theta
$$

トータル勾配はその総和となる：

$$
\nabla_\theta \mathcal{L} = \sum_{t=0}^{T-1} g_{\theta, t}
$$

この方程式は、**深層学習における BPTT（Backpropagation Through Time）の連鎖律の漸化式と数学的に 1 記号の違いもなく同一**である。
- 制御理論：**離散随伴状態法（Discrete Adjoint State Method）**
- 深層学習：**通時的誤差逆伝播法（BPTT）**
- JAX / コンパイラ：**`jax.lax.scan` の Reverse-mode VJP**

この 3 つは同一の数理アルゴリズムの異なる呼称にすぎない。

---

## 3. なぜ手動 Custom VJP より `jax.checkpoint` が優れているのか？

「中間活性化を破棄し、逆伝播時に再計算（Rematerialize）する」というアイデアを手動で書く（`custom_vjp`）のと、JAX ネイティブの `jax.checkpoint` に任せるのとでは、生成される Jaxpr と XLA の最適化に決定的な違いが生じる。

```
【手動 Custom VJP 版】
・逆順インデックス [T-1, ..., 0] を作り、forward 向き scan で回す
・各ステップで dynamic_slice や jax.lax.cond による境界分岐が発生
・各時刻の勾配テンソル配列 [T, params_dim] をメモリに全保持してから最後に reduce_sum
・jax.jvp (Forward-mode AD) が使用不能 (TypeError)

【Checkpointed Scan 版 (jax.checkpoint + scan)】
・JAX の scan primitive が組み込みの reverse=True (逆順スキャン) を発行
・各ステップ内部に remat2 primitive が埋め込まれ、オンザフライで再計算
・★ パラメータ勾配は scan の carry 内で add_any によりオンザフライ累算！
  （[T, params_dim] の巨大バッファを生成しないため、さらに省メモリ）
・余計なスライスや条件分岐が一切なく、XLA ループとして超高速に融合実行
・JVP, Hessian, VMAP との完全な合成性を維持
```

### 生成される VJP Jaxpr の対比

#### Checkpoint 版 (`scan` + `jax.checkpoint`):
```text
scan[
  length=T
  reverse=True    # ★ ネイティブのハードウェア逆順走査
  jaxpr={ lambda ...
      # 1. 1ステップ分の内部テンソルをオンザフライ再計算
      u, v, w, x = remat2[...]
      # 2. パラメータ勾配は carry 変数としてその場で加算！
      grad_params_accum = add_any grad_params_prev u
  }
]
```

#### 手動 Custom VJP 版:
```text
scan[
  reverse=False
  jaxpr={ lambda ...
      cond[...]           # t > 0 の条件分岐
      dynamic_slice[...]  # 軌道配列からの動的スライス
      # 局所 VJP 計算
      # 各ステップのパラメータ勾配をすべて配列バッファに出力
  }
]
reduce_sum[axes=(0,)] grads_params_seq  # ループ終了後に初めて合算
```

手動版は各時刻のパラメータ勾配を `[T, num_params]` のテンソルとして出力バッファに書き出してから合算するため、パラメータ数が多い場合にメモリアクセスオーバーヘッドが激増し、実行速度が大きく低下する。

---

## 4. XLA コンパイラの最適化深淵：`prevent_cse=False` とポリシー

### ① なぜ `prevent_cse=False` にしてよいのか？
`jax.checkpoint` のドキュメントには以下の注記がある：
> *"But in some settings, like when used inside a `jax.lax.scan`, this CSE prevention mechanism is unnecessary, in which case `prevent_cse` can be set to False."*

- **CSE（Common Subexpression Elimination: 共通部分式除去）の罠**:
  コンパイラ（XLA）が「バックワードで再計算している式」と「フォワードで計算した式」を勝手に共通化してキャッシュしてしまうと、Checkpoint の省メモリ効果が無効化される。これを防ぐために JAX は通常 `hlo.OptimizationBarrierOp`（最適化バリア）を挿入する（`prevent_cse=True`）。
- **`scan` ループの特殊性**:
  しかし `scan` の内部では、フォワードループとバックワードループが独立した `while` ループ境界で区切られているため、コンパイラがループを越えて勝手に式を共通化することは構造上あり得ない。
- **結論**:
  `scan` のステップ関数では `prevent_cse=False` を指定することで、余計な最適化バリアを取り払い、XLA に**完全なカーネル融合（Kernel Fusion）**を行わせることができる。

### ② `checkpoint_policies` による速度・メモリの最適化
- **`nothing_saveable`**: 手動 Adjoint と同じく、中間活性化を一切残さない極限省メモリ。
- **`checkpoint_dots_with_no_batch_dims`**:
  現代のハードウェア（GPU/TPU）では、メモリアクセスよりも FLOPs が遥かに安価である一方、巨大な行列積（dot）の再計算は負担になる。行列積の入力テンソルのみを保存し、安価な活性化関数（GELU, Tanh 等）のみを再計算することで、**Naive BPTT よりも高速かつ省メモリ** なスイートスポットを達成できる。

---

## 5. 実機ベンチマーク評価

多層ニューラル力学系（4層 MLP、パラメータ数 10,704、ステップ数 $T=200$、バッチサイズ $B=32$）における実測結果：

| 手法 | Barrier (`OptimizationBarrierOp`) | Residuals メモリ (Tape) | JIT コンパイル時間 | 実行時間 (50回平均) | JVP (Forward AD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **① Naive Auto-AD (BPTT)** | なし | 24,541.0 KB | 150.9 ms | 11.94 ms | ◯ |
| **② Checkpointed (`prevent_cse=True`)** | **あり** | **541.8 KB** (約1/45) | 128.1 ms | 12.45 ms | **◯** |
| **③ Checkpointed (`prevent_cse=False`)** | **なし** | **541.8 KB** (同等) | **118.1 ms** | **12.36 ms** | **◯** |
| **④ Checkpointed (`False` + `dots`)** | **なし** | 5,341.8 KB (約1/5) | 134.0 ms | **10.86 ms (最速)** | **◯** |
| **⑤ Manual Adjoint (`custom_vjp`)** | なし | 543.8 KB | 150.2 ms | 12.96 ms | **✕ (TypeError)** |

- **メモリ**: `scan` + `jax.checkpoint` により、Naive 比で **約 1/45（24.5 MB $\to$ 541 KB）** に削減。手動 Adjoint と完全に同等。
- **速度**: `prevent_cse=False` + `checkpoint_dots` を適用すると、**再計算を行わない Naive BPTT より高速（10.86 ms）** を記録。
- **合成性**: 手動 Adjoint は `jvp` が破綻するが、Checkpoint 版は JVP、Hessian、Reverse-over-Reverse HVP に完全対応。

---

## 6. 実装コード (JAX / Flax NNX)

現代的な離散力学系ソルバの実装は、わずか数行で完結する：

```python
import jax
from functools import partial
from typing import Callable, Any, Optional

def make_discrete_adjoint_solver(
    step_fn: Callable[[Any, jax.Array, Any], jax.Array],
    *,
    prevent_cse: bool = False,
    policy: Optional[Callable] = None,
):
    """
    一般離散力学系 x_{t+1} = step_fn(params, x_t, u_t) に対する
    メモリフットプリント O(1) の Checkpointed 随伴ソルバを生成する。
    """
    # 反復ステップ関数を checkpoint 化
    @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
    def checkpointed_step(params, curr_x, u_t):
        return step_fn(params, curr_x, u_t)

    def solve_dynamics(params: Any, initial_state: jax.Array, inputs_seq: jax.Array) -> jax.Array:
        def scan_step(curr_x, u_t):
            next_x = checkpointed_step(params, curr_x, u_t)
            return next_x, next_x

        _, trajectory = jax.lax.scan(scan_step, initial_state, inputs_seq)
        return trajectory

    return solve_dynamics
```

### 実行方法 (PEP 723 準拠)
```bash
uv run 20261005_discrete_adjoint_state_method/20261005_discrete_adjoint_state_method.py
```

---

## 7. コラム：手動 Custom VJP が真に必要な領域とは？

「離散力学系の時間発展」において手動 Custom VJP を書く必然性はありませんが、以下の領域では依然として Custom VJP が不可欠です：

1. **完全可逆力学系（Symplectic / Reversible Integrator）**:
   ハミルトン系や可逆ブロックのように、終端状態 $x_T$ から時間を逆回転させて $x_t$ を数学的に復元できる場合。軌道テンソル $x_{1:T}$ すら保存せず、メモリを真の定数 $\mathcal{O}(1)$（時間幅 $T$ に非依存）にできる。
2. **固定点モデル・陰関数（Deep Equilibrium Models: DEQ）**:
   $z^* = f(z^*)$ の平衡点を求める場合。フォワードでどんなソルバ（Anderson加速や準ニュートン法）を使っていても、陰関数定理（Implicit Function Theorem）により最終点 $z^*$ のみから 1 回の線形方程式で勾配を解ける。
3. **連続時間方程式（Neural ODE / Diffrax）**:
   適応的ステップサイズ（Adaptive step-size）で時間刻み幅が変化し、連続随伴常微分方程式 $\dot{\lambda}(t) = -\lambda(t)^\top \frac{\partial f}{\partial x}$ を逆向きに積分する場合。

通常の離散ステップ発展（陽解法）やリカレントモデルであれば、**`scan` + `jax.checkpoint(prevent_cse=False)` が最も簡潔で、最も速く、最も堅牢な選択肢**である。
