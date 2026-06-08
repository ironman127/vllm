# GPU 规约算法总结

本文档总结 CUDA kernel 中常见的四种规约算法，结合 vLLM attention 代码中的实际用例进行说明。

---

## 一、算法对比总览

| 算法 | N 的限制 | 结果位置 | 是否需共享内存 | 同步方式 | 适用场景 |
|---|---|---|---|---|---|
| 串行求和 | 任意 | 单线程 | 否 | 无 | 极小规模或调试 |
| 共享内存规约树 | 任意 | thread 0 | 是 | `__syncthreads()` | 跨 warp、任意 N |
| `__shfl_down_sync` | 2 的幂次 | lane 0 | 否 | warp 内自动 | warp 内，只需 lane 0 结果 |
| XOR 蝶形规约 | **2 的幂次** | **所有 lane** | 否 | warp 内自动 | warp 内，所有线程都需要结果 |

---

## 二、串行求和

### 原理

单线程顺序累加，其余线程空闲。

```
初始: a0  a1  a2  a3  a4  a5
step1: a0+a1
step2: a0+a1+a2
step3: a0+a1+a2+a3
step4: a0+a1+a2+a3+a4
step5: a0+a1+a2+a3+a4+a5   ← N-1 步，只有 1 个线程在工作
```

### 示例代码

```cuda
__device__ float serial_sum(float* arr, int N) {
    float sum = 0.f;
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;  // 只有调用线程持有结果
}
```

### 特点

- 时间复杂度：O(N)
- 无并行性，适合 N 极小（如 N<=4）或验证正确性

---

## 三、共享内存规约树

### 原理

将数据对折累加，每轮 stride 减半，用 `__syncthreads()` 保证每轮结束后共享内存一致。

对于非 2 的幂次的 N，令 stride 从「不小于 N 的最小 2 的幂次的一半」开始，
并加入越界保护 `tid + stride < N`。

**以 N=6 为例（stride 从 4 开始）：**

```
初始 smem:   [a0,    a1,    a2, a3, a4, a5]

stride=4     条件 tid+4 < 6，只有 tid=0,1 满足
  tid=0: smem[0] += smem[4]    →  a0+a4
  tid=1: smem[1] += smem[5]    →  a1+a5
             [a0+a4, a1+a5, a2, a3, -, -]

stride=2     条件 tid+2 < 6，tid=0,1 满足
  tid=0: smem[0] += smem[2]    →  a0+a2+a4
  tid=1: smem[1] += smem[3]    →  a1+a3+a5
             [a0+a2+a4, a1+a3+a5, -, -, -, -]

stride=1     条件 tid+1 < 6，tid=0 满足
  tid=0: smem[0] += smem[1]    →  a0+a1+a2+a3+a4+a5  ✓
```

### 示例代码

```cuda
// 任意 N，结果存于 smem[0]
// 调用前：smem[tid] 已写入各线程的值
__device__ void smem_reduce(volatile float* smem, int tid, int N) {
    // 找起始 stride：不小于 N 的最小 2 的幂次的一半
    int stride = 1;
    while (stride < N) stride <<= 1;
    stride >>= 1;

    for (; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < N) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    // smem[0] 持有全局规约结果
}
```

### 特点

- 时间复杂度：O(log N)，步数 = ⌈log₂(N)⌉
- **适用任意 N**，只需加越界保护
- 每步需要 `__syncthreads()`，开销较大
- **结果只在 thread 0**，其他线程需额外广播才能使用

---

## 四、`__shfl_down_sync` 规约

### 原理

同一 warp 内，每个线程将自己的值传给 `lane - delta` 号线程，
每轮 delta 减半，最终 lane 0 累积所有值。

**以 8 线程为例：**

```
初始:         a0   a1   a2   a3   a4   a5   a6   a7

delta=4       每个 lane 加上 lane+4 的值（lane+4 存在才有效）
  lane 0 += lane 4:  a0+a4
  lane 1 += lane 5:  a1+a5
  lane 2 += lane 6:  a2+a6
  lane 3 += lane 7:  a3+a7
              [a0+a4, a1+a5, a2+a6, a3+a7, a4, a5, a6, a7]

delta=2
  lane 0 += lane 2:  a0+a2+a4+a6
  lane 1 += lane 3:  a1+a3+a5+a7
              [a0+a2+a4+a6, a1+a3+a5+a7, ...]

delta=1
  lane 0 += lane 1:  a0+a1+...+a7  ✓
              [全局和, ...]         ← 只有 lane 0 正确
```

### 示例代码

```cuda
// N 必须是 2 的幂次，且 N <= WARP_SIZE
__device__ float shfl_down_reduce(float val, int N) {
    for (int delta = N / 2; delta > 0; delta >>= 1) {
        val += __shfl_down_sync(uint32_t(-1), val, delta);
    }
    return val;  // 只有 lane 0 持有正确结果
}
```

### 特点

- 时间复杂度：O(log N)，无共享内存，无 `__syncthreads()`
- **N 必须是 2 的幂次**
- **结果只在 lane 0**，其他 lane 是中间值
- 适合只需要一个线程持有结果（如 warp 内 reduction 后 lane 0 写回全局内存）

---

## 五、XOR 蝶形规约（Butterfly Reduce）

### 原理

利用 `__shfl_xor_sync`，每轮翻转 lane_id 的一个 bit，
XOR 配对天然保证：A 找 B 的同时 B 也找 A，形成互换对。

经过 log₂(N) 轮后，**所有 lane 都持有完整规约结果**。

**以 8 线程为例：**

```
初始:         a0   a1   a2   a3   a4   a5   a6   a7
lane_id:       0    1    2    3    4    5    6    7

mask=4(100)  翻转 bit2，交换距离 4 的 lane 对
              [a0+a4, a1+a5, a2+a6, a3+a7, a4+a0, a5+a1, a6+a2, a7+a3]

mask=2(010)  翻转 bit1，交换距离 2 的 lane 对
              [a0+a2+a4+a6, a1+a3+a5+a7, (同前两项), ...]

mask=1(001)  翻转 bit0，交换距离 1 的 lane 对
              [SUM, SUM, SUM, SUM, SUM, SUM, SUM, SUM]  ← 所有 lane 都有结果！
```

**XOR 配对的对称性保证（以 mask=4 为例）：**

```
lane 0 (000) XOR 100 = 100 = lane 4  →  lane 0 读 lane 4
lane 4 (100) XOR 100 = 000 = lane 0  →  lane 4 读 lane 0
两者互读，天然形成交换对
```

### 示例代码

```cuda
// N 必须是 2 的幂次，且 N <= WARP_SIZE
// 等价于 vLLM 中的 VLLM_SHFL_XOR_SYNC 宏
__device__ float butterfly_reduce(float val, int N) {
    for (int mask = N / 2; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(uint32_t(-1), val, mask);
    }
    return val;  // 所有 lane 都持有正确结果
}
```

### 与 `__shfl_down_sync` 的关键区别

```
__shfl_down_sync:              __shfl_xor_sync (蝶形):

lane: 0  1  2  3              lane: 0  1  2  3
初始: a  b  c  d              初始: a  b  c  d

delta=2                        mask=2
lane0 += lane2: a+c            lane0 XOR lane2: a+c
lane1 += lane3: b+d            lane1 XOR lane3: b+d
                               lane2 XOR lane0: c+a
                               lane3 XOR lane1: d+b

delta=1                        mask=1
lane0 += lane1: a+b+c+d ✓      lane0 XOR lane1: (a+c)+(b+d) ✓
lane1: b+d (中间值)             lane1 XOR lane0: (b+d)+(a+c) ✓
lane2: c   (未参与)             lane2 XOR lane3: (a+c)+(b+d) ✓
lane3: d   (未参与)             lane3 XOR lane2: (b+d)+(a+c) ✓

结果只在 lane 0                 所有 lane 都有结果
```

### vLLM 中的实际用例

#### 用例 1：Q·K 点积规约（`attention_utils.cuh`）

```cpp
// THREAD_GROUP_SIZE 个线程共同持有一个 Q·K 点积的各维度分量
// 蝶形规约后，组内每个线程都得到完整标量值
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
    using A_vec = typename FloatVec<Vec>::Type;
    A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = vllm::fma(q[ii], k[ii], qk_vec);
    }

    float qk = sum(qk_vec);
    // 蝶形规约：组内所有线程都得到完整的 Q·K 标量
    for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += VLLM_SHFL_XOR_SYNC(qk, mask);  // = __shfl_xor_sync(0xffffffff, qk, mask)
    }
    return qk;
}
```

**调用方（`attention_kernels.cuh`）实际只有 `thread_group_offset == 0` 的 lane 使用 qk：**

```cpp
if (thread_group_offset == 0) {
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;  // 只有 lane 0 写 logits
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);             // 只有 lane 0 更新 qk_max
}
```

因此在这个 kernel 里，用 `__shfl_down_sync`（结果只在 lane 0）同样够用，
代价与蝶形规约完全相同（均为 log₂(T) 次 shuffle）。

**`qk_dot_` 仍然选择蝶形规约的原因：**

`qk_dot_` 是通用工具函数，本身不感知调用方的使用方式。
将结果广播给组内所有 lane 是一种防御性设计，
使其在"所有 lane 都需要结果"的场合下同样可直接复用，无需额外广播。

#### 用例 2：MoE Softmax 中的 max/sum 规约（`topk_softmax_kernels.cu`）

```cpp
// 一行 expert 分数由 THREADS_PER_ROW 个线程共同负责
// 先求行最大值（数值稳定），再求 exp 后的行归一化分母
// 两者都用蝶形规约，使所有线程同步得到相同的 max 和 sum

// max 规约
for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(thread_max,
                     VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
}

// sum 规约
for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
}
// 此后所有线程都持有相同的 thread_max 和 row_sum，可以并行归一化
```

### THREAD_GROUP_SIZE 为何总是 2 的幂次

```cpp
// attention_kernels.cuh 第 133 行
constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
//                                     32   /  {8,16,32,64...}
// WARP_SIZE=32 和 BLOCK_SIZE 均为 2 的幂次
// → THREAD_GROUP_SIZE 天生保证是 2 的幂次，蝶形规约始终正确
```

---

## 六、非 2 的幂次下蝶形规约的错误示例

以 N=6 为例，说明为什么不能直接用 XOR 蝶形：

```
初始: [a0, a1, a2, a3, a4, a5]  lane 0~5

mask=3(011)
  lane 4 (100) XOR 011 = 111 = lane 7  →  lane 4 读 lane 7（越界，垃圾值！）
  lane 5 (101) XOR 011 = 110 = lane 6  →  lane 5 读 lane 6（越界，垃圾值！）

mask=1(001)
  lane 0~3 得到 a0+a1+a2+a3    ← a4、a5 永远未被累加！
  lane 4~5 得到垃圾值
```

**修复方案**：补零到 2 的幂次（N=8，令 smem[6]=smem[7]=0），
XOR 蝶形自动处理，结果仍然正确。

---

## 七、选择指南

```
需要规约的数据在哪里？
│
├─ 在单个 warp 内（N ≤ 32）
│   │
│   ├─ N 是 2 的幂次？
│   │   ├─ 是
│   │   │   ├─ 所有线程都需要结果？  → XOR 蝶形规约（__shfl_xor_sync）
│   │   │   └─ 只有 lane 0 需要？   → __shfl_down_sync 规约
│   │   └─ 否
│   │       └─ 补零到 2 的幂次后用 XOR 蝶形，或用共享内存规约树
│   │
└─ 跨多个 warp 或 N 很大
    └─ 共享内存规约树（搭配 warp 内 __shfl_down_sync 做两阶段规约）
```
