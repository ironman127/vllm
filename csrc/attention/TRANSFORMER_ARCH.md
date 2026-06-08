# Decode-Only Transformer 完整架构图与计算公式

以 GPT 系列 / LLaMA 系列为代表的 Causal (Decode-Only) Transformer。

---

## 一、整体架构图

```
输入 Token 序列： [x_1, x_2, ..., x_T]
                         │
                         ▼
┌─────────────────────────────────────────┐
│           Token Embedding               │
│   E ∈ R^{V×d}  →  X ∈ R^{T×d}         │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│        Positional Encoding              │
│   X ← X + PE  （绝对位置）              │
│   或  RoPE 在 Attention 内部施加        │
└─────────────────────┬───────────────────┘
                      │
          ┌───────────┘
          │   重复 N 次（Transformer Block）
          ▼
┌─────────────────────────────────────────────────────┐
│                  Transformer Block × N               │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │             Pre-Norm（RMSNorm）              │   │
│   └───────────────────┬─────────────────────────┘   │
│                       │                             │
│   ┌───────────────────▼─────────────────────────┐   │
│   │         Causal Multi-Head Attention          │   │
│   │                                             │   │
│   │   Q=XW_Q   K=XW_K   V=XW_V                 │   │
│   │        │        │        │                  │   │
│   │   ┌────▼────────▼────────▼────┐             │   │
│   │   │  Split into H heads        │             │   │
│   │   └────────────┬──────────────┘             │   │
│   │                │  （每个 head）               │   │
│   │   ┌────────────▼──────────────┐             │   │
│   │   │   RoPE(Q_i)  RoPE(K_i)   │             │   │
│   │   └────┬──────────┬───────────┘             │   │
│   │        │          │                         │   │
│   │   ┌────▼──────────▼───────────┐             │   │
│   │   │  Scaled Dot-Product Attn  │             │   │
│   │   │  + Causal Mask            │             │   │
│   │   └────────────┬──────────────┘             │   │
│   │                │                            │   │
│   │   ┌────────────▼──────────────┐             │   │
│   │   │    Concat H heads         │             │   │
│   │   └────────────┬──────────────┘             │   │
│   │                │                            │   │
│   │   ┌────────────▼──────────────┐             │   │
│   │   │    Output Proj  W_O       │             │   │
│   │   └────────────┬──────────────┘             │   │
│   └────────────────┼────────────────────────────┘   │
│                    │                                 │
│              ┌─────▼──────┐                         │
│              │  + Residual │  X ← X + Attn_out       │
│              └─────┬──────┘                         │
│                    │                                 │
│   ┌────────────────▼────────────────────────────┐   │
│   │             Pre-Norm（RMSNorm）              │   │
│   └───────────────────┬─────────────────────────┘   │
│                       │                             │
│   ┌───────────────────▼─────────────────────────┐   │
│   │          Feed-Forward Network (FFN)          │   │
│   │                                             │   │
│   │   经典 FFN：Linear → GELU → Linear           │   │
│   │   SwiGLU  ：Gate × SiLU(Up) → Linear        │   │
│   │                                             │   │
│   └───────────────────┬─────────────────────────┘   │
│                       │                             │
│              ┌────────▼────────┐                    │
│              │   + Residual    │  X ← X + FFN_out    │
│              └────────┬────────┘                    │
└───────────────────────┼─────────────────────────────┘
          │   结束 N 次循环
          ▼
┌─────────────────────────────────────────┐
│           Final RMSNorm                 │
└─────────────────────┬───────────────────┘
                      │
┌─────────────────────▼───────────────────┐
│         LM Head（Linear，共享 Embedding）│
│         logits ∈ R^{T×V}               │
└─────────────────────┬───────────────────┘
                      │
┌─────────────────────▼───────────────────┐
│         Softmax → 下一 token 概率        │
└─────────────────────────────────────────┘
```

---

## 二、各层详细计算公式

### 2.1 Token Embedding

$$
X = \text{Embedding}(\text{tokens}) \in \mathbb{R}^{T \times d}
$$

- $T$：当前序列长度
- $d$：模型隐层维度（hidden size）
- Embedding 矩阵 $E \in \mathbb{R}^{V \times d}$，$V$ 为词表大小

---

### 2.2 位置编码

#### 绝对位置编码（Sinusoidal / Learned，GPT-2 风格）

$$
\text{PE}(t, 2i)   = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad
\text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

$$
X \leftarrow X + \text{PE}
$$

#### RoPE（Rotary Position Embedding，LLaMA 风格）

对 Q、K 中每对相邻维度 $(2i, 2i+1)$ 施加旋转：

$$
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix}
=
\begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix},
\quad \theta_i = \frac{1}{10000^{2i/d_h}}
$$

其中 $t$ 为位置索引，$d_h = d/H$ 为每个 head 的维度。

---

### 2.3 RMSNorm（Pre-Norm，每个 Block 入口施加）

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma,
\quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

- $\gamma \in \mathbb{R}^d$：可学习缩放参数
- $\epsilon$：数值稳定项（如 $10^{-6}$）

（LayerNorm 额外减去均值 $\mu$，RMSNorm 省略此步骤）

---

### 2.4 Multi-Head Causal Self-Attention

#### (1) 线性投影

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d},\quad Q, K, V \in \mathbb{R}^{T \times d}
$$

#### (2) 拆分多头

$$
Q = [Q_1, Q_2, \ldots, Q_H],\quad Q_i \in \mathbb{R}^{T \times d_h},\quad d_h = d / H
$$

（K、V 同理；若使用 GQA，$K$、$V$ 的头数为 $H_{kv} < H$）

#### (3) 施加 RoPE（如使用）

$$
Q_i \leftarrow \text{RoPE}(Q_i),\quad K_i \leftarrow \text{RoPE}(K_i)
$$

#### (4) Scaled Dot-Product Attention（含因果掩码）

$$
\text{score}_{i} = \frac{Q_i K_i^\top}{\sqrt{d_h}} \in \mathbb{R}^{T \times T}
$$

施加**因果掩码**（下三角），屏蔽未来 token：

$$
\text{score}_{i}[t, s] =
\begin{cases}
\dfrac{q_t \cdot k_s}{\sqrt{d_h}} & s \leq t \\[6pt]
-\infty & s > t
\end{cases}
$$

Softmax 归一化：

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\displaystyle\sum_{k=1}^{n} e^{z_k}}
$$

将每行 score 向量归一化为概率分布（所有元素非负且和为 1）：

$$
A_i[t, s] = \frac{\exp\!\left(\text{score}_i[t, s]\right)}{\displaystyle\sum_{s'=1}^{t} \exp\!\left(\text{score}_i[t, s']\right)} \in \mathbb{R}^{T \times T}
$$

注意：因果掩码将 $s > t$ 的位置置为 $-\infty$，$e^{-\infty} = 0$，
故分母只对 $s \leq t$ 的位置求和，保证未来 token 权重为 0。每个输出都是当前位置对前面位置的value注意力权重之和。

数值稳定版本（实际实现，减去行最大值防止 overflow）：

$$
A_i[t, s] = \frac{\exp\!\left(\text{score}_i[t, s] - m_t\right)}{\displaystyle\sum_{s' \leq t} \exp\!\left(\text{score}_i[t, s'] - m_t\right)},
\quad m_t = \max_{s' \leq t}\, \text{score}_i[t, s']
$$

加权求和：

$$
O_i = A_i V_i \in \mathbb{R}^{T \times d_h}
$$

#### (5) 合并多头 + 输出投影

$$
O = \text{Concat}(O_1, O_2, \ldots, O_H) \in \mathbb{R}^{T \times d}
$$

$$
\text{Attn\_out} = O W_O,\quad W_O \in \mathbb{R}^{d \times d}
$$

#### (6) 残差连接

$$
X \leftarrow X + \text{Attn\_out}
$$

---

### 2.5 GQA / MQA（分组查询注意力，可选）

GQA 将 $H$ 个 Query head 分成 $G$ 组，每组共享一对 K/V head（$H_{kv} = G$）：

$$
K_i = K_{\lfloor i \cdot G / H \rfloor},\quad V_i = V_{\lfloor i \cdot G / H \rfloor}
$$

- MQA（Multi-Query Attention）：$G = 1$，所有 Q head 共享同一 K/V
- GQA：$1 < G < H$，折中方案（LLaMA-2/3 采用）

投影矩阵维度变为：

$$
W_K, W_V \in \mathbb{R}^{d \times (d_h \cdot G)}
$$

---

### 2.6 Feed-Forward Network (FFN)

#### 经典 FFN（GPT-2 风格）

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

$$
W_1 \in \mathbb{R}^{d \times d_{ff}},\quad W_2 \in \mathbb{R}^{d_{ff} \times d},\quad d_{ff} = 4d
$$

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702 x)
$$

#### SwiGLU（LLaMA 风格）

$$
\text{FFN}_{\text{SwiGLU}}(x) = W_{\text{down}} \cdot \bigl(\text{SiLU}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x\bigr)
$$

$$
W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}'},\quad
W_{\text{down}} \in \mathbb{R}^{d_{ff}' \times d},\quad
d_{ff}' = \tfrac{2}{3} \times 4d
$$

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

残差连接：

$$
X \leftarrow X + \text{FFN\_out}
$$

---

### 2.7 Final RMSNorm

$$
X \leftarrow \text{RMSNorm}(X)
$$

---

### 2.8 LM Head + Softmax

$$
\text{logits} = X W_E^\top \in \mathbb{R}^{T \times V}
$$

（$W_E$ 与 Token Embedding 矩阵共享权重）

生成阶段只取最后一个位置 $T$ 的 logits，经温度缩放后做 softmax：

$$
P(\text{next token} = v) = \frac{\exp\!\left(\text{logits}[T, v] / \tau\right)}{\displaystyle\sum_{v'=1}^{V} \exp\!\left(\text{logits}[T, v'] / \tau\right)}
$$

- $\tau$：温度参数，$\tau \to 0$ 趋近 argmax（贪心），$\tau > 1$ 分布更平坦（更随机）
- 输出为词表上的概率分布 $P \in \mathbb{R}^V$，$\sum_v P_v = 1$

---

## 三、完整 Transformer Block 公式汇总

设第 $\ell$ 层输入为 $X^{(\ell)}$，则：

$$
\boxed{
\begin{aligned}
&\tilde{X}^{(\ell)}  &&= \text{RMSNorm}\!\left(X^{(\ell)}\right) \\[4pt]
&\text{Attn}^{(\ell)} &&= \text{MultiHeadAttn}\!\left(\tilde{X}^{(\ell)},\, \tilde{X}^{(\ell)},\, \tilde{X}^{(\ell)}\right) \\[4pt]
&X'^{(\ell)}         &&= X^{(\ell)} + \text{Attn}^{(\ell)} \\[4pt]
&\tilde{X}'^{(\ell)} &&= \text{RMSNorm}\!\left(X'^{(\ell)}\right) \\[4pt]
&\text{FFN}^{(\ell)} &&= W_{\text{down}}\!\left(\text{SiLU}(W_{\text{gate}}\,\tilde{X}'^{(\ell)}) \odot W_{\text{up}}\,\tilde{X}'^{(\ell)}\right) \\[4pt]
&X^{(\ell+1)}        &&= X'^{(\ell)} + \text{FFN}^{(\ell)}
\end{aligned}
}
$$

---

## 四、Decode 阶段 KV Cache 加速

推理生成阶段（每步只生成 1 个新 token），无需重新计算历史 K/V：

```
Prefill 阶段（处理 prompt，长度 T_p）：
  计算并缓存所有 token 的 K, V → KV Cache

Decode 阶段（每步生成 1 个 token）：
  新 token x_t 只计算 Q_t, K_t, V_t
  K_t, V_t 追加入 KV Cache
  Attention 计算：
    Q_t ∈ R^{1×d_h}，K_cache ∈ R^{(T_p+t)×d_h}

  score_t = Q_t · K_cache^T / sqrt(d_h)   ∈ R^{1×(T_p+t)}
  a_t     = softmax(score_t)               ∈ R^{1×(T_p+t)}
  o_t     = a_t · V_cache                  ∈ R^{1×d_h}
```

vLLM 用 **Paged Attention** 管理 KV Cache（分页存储，非连续内存），
对应本目录中 `paged_attention_v2.cu` / `attention_kernels.cuh` 的实现。

---

## 五、主要超参数一览

| 符号 | 含义 | 典型值（LLaMA-3 8B）|
|---|---|---|
| $d$ | Hidden size | 4096 |
| $H$ | Q head 数 | 32 |
| $H_{kv}$ | KV head 数（GQA）| 8 |
| $d_h = d/H$ | 每个 head 的维度 | 128 |
| $d_{ff}'$ | FFN 中间维度（SwiGLU）| 14336 |
| $N$ | Transformer block 层数 | 32 |
| $V$ | 词表大小 | 128256 |
| $T_{\max}$ | 最大序列长度 | 8192 |

---

## 六、架构变体对比

| 特性 | GPT-2 | LLaMA-2 | LLaMA-3 |
|---|---|---|---|
| 位置编码 | Learned Abs | RoPE | RoPE |
| Norm 类型 | LayerNorm（Post）| RMSNorm（Pre）| RMSNorm（Pre）|
| FFN 激活 | GELU | SwiGLU | SwiGLU |
| Attention | MHA | GQA | GQA |
| Norm 位置 | Post-Norm | Pre-Norm | Pre-Norm |
| Bias | 有 | 无 | 无 |
