# Paged Attention V2 实现说明

本文档总结 `paged_attention_v2` 的设计思路与代码结构，供阅读 `attention_kernels.cuh` /
`paged_attention_v2.cu` 时参考。

---

## 一、文件结构

| 文件 | 职责 |
|---|---|
| `paged_attention_v2.cu` | Kernel 启动入口：设置 grid/block、分配 shared mem、调用下面两个 kernel |
| `attention_kernels.cuh` | 两个 kernel 的具体实现（主 kernel + reduce kernel）|
| `attention_utils.cuh` | `Qk_dot`、`block_sum` 等工具函数 |
| `attention_dtypes.h` | `Vec`、`FloatVec`、`dot`、`from_float` 等向量类型辅助 |

---

## 二、整体架构：两个 Kernel

```
paged_attention_v2_kernel          paged_attention_v2_reduce_kernel
（主 kernel：并行计算各 partition）  （reduce kernel：合并所有 partition）
         |                                       |
  tmp_out / exp_sums / max_logits    -->    最终 out
```

### 为什么需要两个 Kernel？

Softmax 依赖全局归一化项，无法直接拆分。V2 将序列沿时间维度切成多个
**partition**（每片 `PARTITION_SIZE=512` tokens），各 partition 并行计算**局部 softmax**，
再由 reduce kernel 用 **online softmax（log-sum-exp）** 合并，保证与单次完整
softmax 数学等价，同时大幅提升长序列时的 GPU 并行度。

---

## 三、关键编译期常量

```cpp
NUM_THREADS    = 128   // 每个 CUDA block 的线程数
PARTITION_SIZE = 512   // 每个 partition 处理的 token 数
BLOCK_SIZE     = 16    // KV cache 每个物理块的 token 数（Paged Attention 的"页"）
WARP_SIZE      = 32

// 派生常量
NUM_WARPS         = NUM_THREADS / WARP_SIZE           = 4
THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE            = 2  // 协作计算一个 QK 点积的线程数
NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE   = 64
```

---

## 四、CUDA Grid 设计

```cpp
// paged_attention_v2.cu 第 82、87 行
int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
dim3 grid(num_heads, num_seqs, max_num_partitions);
//   x 轴       y 轴         z 轴
```

每个 CUDA block 独立负责一个 **(head, seq, partition)** 三元组。

---

## 五、并行划分层次

主 kernel 的工作按四个层次逐级分配：

```
CUDA Block（对应一个 partition）
  │
  │  一个 CUDA block 负责：
  │    (head_idx, seq_idx, partition_idx) 三元组
  │    即：某条序列的某个 attention head 的 512 个 token 分片
  │
  ├─► Warp（对应一个 KV block，每轮循环）
  │     4 个 warp 跨步分配 partition 内的 32 个 KV block：
  │       warp0 → block 0, 4, 8, ..., 28
  │       warp1 → block 1, 5, 9, ..., 29
  │       warp2 → block 2, 6, 10, ..., 30
  │       warp3 → block 3, 7, 11, ..., 31
  │
  └─► Thread Group（对应一个 token）
        每个 warp 内划分为 BLOCK_SIZE = 16 个 thread group（每 token 对应一个）
        每个 thread group 由 THREAD_GROUP_SIZE = WARP_SIZE/BLOCK_SIZE = 2 个线程组成
        负责计算一个 token 的完整 dot(Q, K)：
          线程0：读 K 的偶数向量组，累加 partial_sum
          线程1：读 K 的奇数向量组，累加 partial_sum
          → warp shuffle 规约得到完整 QK 标量
```

### 共享内存 vs 寄存器 的分工

```
Q（query 向量）：
  所有 warp 共享同一个 Q（同一 token 的 query）
  → 加载到 shared memory，读一次，全 block 复用
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD]

K（key 向量）：
  每个 warp 读取不同 KV block 的 K，warp 之间无复用
  → 直接加载到寄存器（局部变量），用完即弃
  K_vec k_vecs[NUM_VECS_PER_THREAD]   // 寄存器

logits（QK 得分）：
  所有 warp 的点积结果汇总
  → 写入 shared memory，供 softmax 和 AV 乘法使用
  __shared__ float logits[PARTITION_SIZE]
```

---

## 六、K/V Cache 内存布局设计

### K Cache：`[num_blocks, num_kv_heads, head_size/x, block_size, x]`

`x = 16 / sizeof(T)`，fp16 时 x=8，即每次读取 8 个元素 = 16 字节（128-bit）。

**为什么不用 `[block_size, head_size]`（按 token 顺序存）？**

计算 QK 时，warp 内 16 个 thread group 同时读同一 KV block 里 16 个不同 token 的第 i 段 head 数据：

```
[block_size, head_size] 布局（AoS）：
  token 0 的第 i 段 → 地址 base + i
  token 1 的第 i 段 → 地址 base + head_size + i   ← 相差 256 字节
  → 16 个线程地址间隔 256 字节 → 16 条 cache line → 有效利用率 6.25%

[head_size/x, block_size, x] 布局（SoA）：
  token 0 的第 i 段 → 地址 base + i * block_size * x + 0
  token 1 的第 i 段 → 地址 base + i * block_size * x + x   ← 紧邻
  → 16 个线程地址连续 → 1 条 cache line → 有效利用率 100%
```

内层 `x` 维度的作用：把 x 个相邻 head 元素打包，单线程一条 `ld.global.v4.u32`（128-bit）全部读出。

| 维度 | 优化目标 |
|---|---|
| 外层 `[head_size/x, block_size]` | **Warp Coalescing**：16 个线程的地址连续，1 条 cache line |
| 内层 `[x]` | **向量化**：单线程 128-bit 宽指令，减少指令条数 |

---

### V Cache：`[num_blocks, num_kv_heads, head_size, block_size]`

计算 AV 时访问模式不同：**每个线程固定负责若干 head_dim 行，顺序遍历 block 内所有 token**。

```
对于 head_dim 行 h、block 内 token t：
  地址 = h * block_size + t
  → 单线程按 t=0,1,2,...递增，地址连续 → 顺序读取，L1 cache 友好
```

不需要 x 分组，因为每个线程按行顺序读，天然连续，不存在 K 那样的跨 token 跳跃问题。

---

### K vs V 布局对比

| | K Cache | V Cache |
|---|---|---|
| 布局 | `[head_size/x, block_size, x]` | `[head_size, block_size]` |
| 访问方向 | warp 内多线程同时读不同 token（跨行）| 单线程顺序读一行的所有 token |
| 关键优化 | Warp coalescing + 向量化 | 顺序访问（无需 coalescing）|
| x 分组 | 需要 | 不需要 |

---

## 七、主 Kernel 执行流程（`paged_attention_kernel`）


### Step 1：确定工作范围

```cpp
partition_idx = blockIdx.z
start_token   = partition_idx * PARTITION_SIZE
end_token     = min(start_token + PARTITION_SIZE, seq_len)
start_block   = start_token / BLOCK_SIZE   // KV 物理块起始索引
end_block     = end_token   / BLOCK_SIZE
```

若 partition 超出序列长度，直接 early exit。

### Step 2：加载 Q 到 Shared Memory

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

所有 warp 共享同一份 Q，避免重复从 global memory 读取。

### Step 3：计算 QK^T → logits

```
Shared Memory: float logits[PARTITION_SIZE]

for block_idx in [start_block, end_block), 步长 = NUM_WARPS:
    每个 warp 负责一个 KV block（按 warp_idx 跨步分配）
        通过 block_table 查找物理块地址（Paged Attention 核心）
        thread group 内协作向量化读取 K，计算 dot(Q, K) * scale
        将 QK 得分写入 logits[token_idx - start_token]
        维护 warp 内局部 qk_max
```

**K cache 内存布局**：`[num_blocks, num_kv_heads, head_size/x, block_size, x]`
其中 `x = 16 / sizeof(T)`，交错存储，支持 16 字节向量化读取。

### Step 4：Softmax（两级 Reduction）

```
1. Warp 内 reduction（shfl_xor）→ 每个 warp 的 qk_max
2. Block 内 reduction（red_smem）→ 全局 qk_max，broadcast 给所有线程

logits[i]  = exp(logits[i] - qk_max)      // 数值稳定
exp_sum    = block_sum(logits[i])          // 跨 warp reduce
logits[i] /= exp_sum                       // 归一化，原地覆写
```

若启用 partitioning，将 `qk_max` 和 `exp_sum` 写入全局内存：

```cpp
max_logits[seq, head, partition] = qk_max
exp_sums  [seq, head, partition] = exp_sum
```

### Step 5：计算 Attention·V → 累加输出

```
float accs[NUM_ROWS_PER_THREAD]   // FP32 累加器，驻留寄存器

for block_idx in [start_block, end_block), 步长 = NUM_WARPS:
    每个 warp 处理一个 KV block：
        读 logits[token_idx]（已归一化的 softmax 权重）
        通过 block_table 查 v_cache 物理地址
        accs[i] += dot(logits_vec, v_vec)
```

**V cache 内存布局**：`[num_blocks, num_kv_heads, head_size, block_size]`
行为 head_dim，列为 block_size，便于按权重加权累加。
（与 K cache 布局不同，两者分别针对各自访问模式优化）

### Step 6：跨 Warp Reduction → 写出 tmp_out

```
树形 reduce（log2(NUM_WARPS) 轮）：
    高编号 warp 写 accs 到 out_smem（复用 logits 的 shared mem 空间）
    低编号 warp 读并累加
最终 warp 0 写入：
    tmp_out[seq, head, partition, :] = accs（转回 fp16/bf16）
```

---

## 八、Reduce Kernel（`paged_attention_v2_reduce_kernel`）

Grid = `(num_heads, num_seqs)`，每个 block 合并一个 (head, seq) 的所有 partition。

### 数学推导（Online Softmax）

设第 $p$ 个 partition 的局部结果：

$$m_p = \text{local\_max},\quad s_p = \sum_i e^{x_i - m_p},\quad o_p = \frac{\sum_i e^{x_i - m_p} \cdot V_i}{s_p}$$

合并步骤：

$$m^* = \max_p(m_p)$$
$$\hat{s}_p = s_p \cdot e^{m_p - m^*} \quad\text{（rescale 对齐全局 max）}$$
$$\text{output} = \frac{\sum_p \hat{s}_p \cdot o_p}{\sum_p \hat{s}_p}$$

与单次完整 softmax **数学等价**，数值稳定。

```cpp
// 对应代码（attention_kernels.cuh）
float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);  // rescale
acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
```

---

## 九、完整数据流

```
输入
  Q:           [num_seqs, num_heads, head_size]
  k_cache:     [num_blocks, num_kv_heads, head_size/x, block_size, x]
  v_cache:     [num_blocks, num_kv_heads, head_size, block_size]
  block_table: [num_seqs, max_num_blocks_per_seq]
        |
        v  主 kernel（并行 x max_num_partitions）
中间结果
  tmp_out:     [num_seqs, num_heads, max_num_partitions, head_size]
  exp_sums:    [num_seqs, num_heads, max_num_partitions]
  max_logits:  [num_seqs, num_heads, max_num_partitions]
        |
        v  reduce kernel（每个 head x seq 一个 block）
输出
  out:         [num_seqs, num_heads, head_size]
```

---

## 十、设计亮点

| 设计 | 解决的问题 |
|---|---|
| **Partition（z 轴并行）** | 长序列时 SM 利用率不足 |
| **Block Table 间接寻址** | KV cache 非连续存储（分页内存管理）|
| **Thread Group 协作点积** | QK 计算向量化，提升访存效率 |
| **两级 Reduction（warp→block）** | 高效跨线程归约，避免 atomic 竞争 |
| **Shared mem 复用（logits→out_smem）** | 节省 shared memory 用量 |
| **FP32 累加器** | 混合精度计算的数值精度保障 |
| **Online Softmax（log-sum-exp）** | partition 合并时数值稳定 |
| **K/V 不同内存布局** | K 优化向量化读取；V 优化按行加权累加 |
| **`__restrict__` + `const`** | 触发只读缓存（ldg），提升 KV cache 读取带宽 |

---

## 十一、V1 vs V2 对比

| | V1 | V2 |
|---|---|---|
| Grid z 维 | 1 | `max_num_partitions` |
| 长序列并行度 | 低（单 block 串行扫全序列）| 高（多 block 并行）|
| 中间缓冲区 | 不需要 | 需要（`tmp_out / exp_sums / max_logits`）|
| 核心技巧 | 标准 softmax | Online Softmax（log-sum-exp）|
| 适用场景 | 短序列 | 长序列 |
