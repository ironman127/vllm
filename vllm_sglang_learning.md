# 深入学习 vLLM 与 SGLang 指南

> 面向有一定 PyTorch / Transformer 基础，希望深入理解推理引擎原理与源码的工程师。

---

## 目录

1. [前置知识](#1-前置知识)
2. [vLLM 深入学习路径](#2-vllm-深入学习路径)
   - 2.1 核心概念
   - 2.2 源码阅读顺序
   - 2.3 关键模块精讲
3. [SGLang 深入学习路径](#3-sglang-深入学习路径)
   - 3.1 核心概念
   - 3.2 源码阅读顺序
   - 3.3 关键模块精讲
4. [vLLM vs SGLang 横向对比](#4-vllm-vs-sglang-横向对比)
5. [动手实验清单](#5-动手实验清单)
6. [推荐资料](#6-推荐资料)

---

## 1. 前置知识

在深入源码前，需要先打牢以下基础：

| 领域 | 具体内容 |
|------|---------|
| Transformer 推理 | Attention 计算流程、KV Cache 原理、自回归生成过程 |
| GPU 内存模型 | HBM 带宽瓶颈、显存碎片、CUDA 内存分配 |
| 分布式推理 | Tensor Parallelism、Pipeline Parallelism 基本原理 |
| Python 异步 | `asyncio`、ZMQ 进程间通信 |
| CUDA 算子 | FlashAttention 基本思想（不需要手写，但要理解其作用） |

**建议先读：**
- 《Attention Is All You Need》
- Hugging Face Transformers 的 `generate()` 流程源码
- FlashAttention 论文摘要部分

---

## 2. vLLM 深入学习路径

**仓库地址：** https://github.com/vllm-project/vllm

### 2.1 核心概念

#### PagedAttention

vLLM 最核心的创新，受操作系统虚拟内存管理启发：

- KV Cache 被切分成固定大小的 **Block**（默认 16 tokens/block）
- 通过 **BlockTable** 映射逻辑块与物理块，允许非连续内存分配
- 显存浪费率从 60~80% 降至 **< 4%**
- 支持 **Prefix Sharing**：多个请求共享相同前缀（如 system prompt）的物理块，零拷贝复用

```
逻辑序列: [tok0, tok1, ..., tok31]
              ↓ BlockTable 映射
物理块:   Block#5 [tok0..tok15] → Block#12 [tok16..tok31]
```

#### Continuous Batching（迭代级调度）

- 传统方法：一批请求同时开始、同时结束（静态 batch）
- vLLM：每个 decode step 后，完成的请求立即释放，新请求立即加入
- 显著提升 GPU 利用率，吞吐提升 **2~24×**

#### Chunked Prefill

- 长 prompt 拆成多个 chunk（如每步 2K tokens）分批 prefill
- 避免单个长请求独占 GPU，防止 decode 请求"饥饿"

#### Automatic Prefix Caching (APC)

- 链式哈希标识每个 block 的内容
- 相同前缀的请求命中缓存，跳过 prefill 计算
- 多轮对话、RAG 场景收益显著，重复前缀 prefill 计算减少最高 **98%**

---

### 2.2 源码阅读顺序

#### 阶段一：入口与请求生命周期（1~2 天）

```
vllm/
├── entrypoints/
│   ├── openai/api_server.py   ← HTTP API 入口，理解请求如何进来
│   └── llm.py                 ← 离线推理入口 LLM.generate()
├── engine/
│   ├── llm_engine.py          ← 核心引擎，串联所有组件
│   └── async_llm_engine.py    ← 异步版本，生产部署用
```

跟踪一个请求从 `POST /v1/completions` 到返回 token 的完整路径。

#### 阶段二：调度器（2~3 天）

```
vllm/core/
├── scheduler.py               ← 调度核心：prefill/decode 队列管理
├── block_manager.py           ← 物理块分配与回收
└── policy.py                  ← 调度策略（FCFS 等）
```

重点理解：
- `SchedulerOutput` 包含哪些信息
- `_schedule_prefills()` 和 `_schedule_running()` 的逻辑差异
- Block 如何被 allocate / free / copy-on-write

#### 阶段三：KV Cache 管理（2 天）

```
vllm/core/
└── block/                     ← Block 抽象层
    ├── interfaces.py
    ├── cpu_gpu_block_allocator.py
    └── prefix_caching_block.py  ← APC 实现
```

画出 `BlockTable` 的数据结构图，手动模拟一次 block 分配过程。

#### 阶段四：模型执行（2~3 天）

```
vllm/worker/
├── worker.py                  ← GPU worker，执行前向传播
├── model_runner.py            ← 构造输入张量、调用模型
└── cache_engine.py            ← KV Cache 的实际 GPU 内存管理

vllm/attention/
├── backends/
│   ├── flash_attn.py          ← FlashAttention 后端
│   └── xformers.py
└── ops/paged_attn.py          ← PagedAttention CUDA 算子封装
```

#### 阶段五：分布式（选修，2~3 天）

```
vllm/distributed/
├── parallel_state.py          ← 进程组管理
└── communication_op.py        ← all-reduce、broadcast 封装

vllm/model_executor/models/   ← 各模型的 TP 切分实现
```

---

### 2.3 关键模块精讲

#### Scheduler 调度循环

```python
# scheduler.py 核心逻辑（简化）
def schedule(self):
    # 1. 尝试将 waiting 队列中的请求移入 running（prefill）
    scheduled_prefills = self._schedule_prefills()
    
    # 2. 处理 running 队列中的请求（decode），必要时 preempt
    scheduled_running = self._schedule_running()
    
    # 3. 返回本步需要执行的请求集合
    return SchedulerOutput(scheduled_prefills, scheduled_running, ...)
```

关键状态机：`waiting → running → swapped → running → finished`

#### BlockManager 分配策略

- `allocate(seq_group)`：为新请求分配物理块
- `can_allocate(seq_group)`：检查显存是否足够（用于调度决策）
- `free(seq)`：释放请求占用的所有块
- `fork(parent_seq, child_seq)`：beam search 时 copy-on-write

---

## 3. SGLang 深入学习路径

**仓库地址：** https://github.com/sgl-project/sglang

### 3.1 核心概念

#### RadixAttention

SGLang 最核心的创新：用 **Radix Tree（前缀树）** 管理 KV Cache。

```
RadixTree 示例：
root
├── "你好，请帮我" → Node A（KV Cache blocks: [3,7]）
│   ├── "写一首诗" → Node B（blocks: [11]）
│   └── "翻译这段话" → Node C（blocks: [15]）
└── "今天天气" → Node D（blocks: [2]）
```

- 不同请求的相同前缀**自动复用**同一批物理 KV Cache blocks
- LRU 淘汰：内存不足时驱逐最近最少使用的节点
- 相比 vLLM 的 APC，RadixAttention 粒度更细，缓存命中率提升 **3~5×**

#### 多进程架构（ZMQ IPC）

```
Main Process
├── HTTP/gRPC Server
├── TokenizerManager
└── Engine
      ↕ ZMQ
Scheduler Process
├── scheduler.py            ← 调度决策
├── RadixCache              ← KV Cache 树
└── ModelExecutor
      ↕ NCCL/ZMQ
DetokenizerManager Process  ← 异步 detokenize，不阻塞推理
```

#### 结构化输出（Constrained Decoding）

- 支持 JSON Schema、正则表达式约束生成
- 通过有限状态机在每步解码时限制合法 token 集合
- 零额外推理开销（mask 在 logits 层完成）

---

### 3.2 源码阅读顺序

#### 阶段一：整体架构（1 天）

```
python/sglang/srt/
├── server.py                  ← 启动入口，理解多进程如何拉起
├── managers/
│   ├── tokenizer_manager.py   ← 请求预处理、tokenize
│   └── detokenizer_manager.py ← 异步解码输出
└── entrypoints/
    └── http_server.py         ← FastAPI 路由
```

#### 阶段二：RadixCache（2~3 天）

```
python/sglang/srt/mem_cache/
├── radix_cache.py             ← 核心：RadixTree 实现
│   ├── TreeNode               ← 节点：children, value(块索引), lock_ref, last_access_time
│   ├── RadixCache.match_prefix()   ← 前缀匹配，找最长可复用前缀
│   ├── RadixCache.cache_finished_req()  ← 请求完成后插入树
│   └── RadixCache._evict_lru()     ← LRU 淘汰
└── base_prefix_cache.py       ← 抽象基类
```

手动在纸上画出多个请求插入/命中 RadixTree 的过程，是理解这块的最快方法。

#### 阶段三：Scheduler（2 天）

```
python/sglang/srt/managers/
└── scheduler.py
    ├── add_request()          ← 新请求入队
    ├── schedule()             ← 核心调度循环
    └── process_batch()        ← 构造本步的执行 batch
```

对比 vLLM 的 scheduler 理解两者设计哲学差异。

#### 阶段四：模型执行与 Attention 后端（2~3 天）

```
python/sglang/srt/model_executor/
├── model_runner.py            ← forward pass 封装
└── cuda_graph_runner.py       ← CUDAGraph 优化（减少 kernel launch 开销）

python/sglang/srt/layers/attention/
├── triton_ops/                ← Triton 实现的 attention 算子
├── flashinfer_backend.py      ← FlashInfer 后端（性能最优）
└── radix_attention.py         ← RadixAttention 与 attention 后端的桥接
```

#### 阶段五：结构化输出（选修，1~2 天）

```
python/sglang/srt/constrained/
├── jump_forward.py            ← Jump-Forward Decoding（跳过确定性 token）
└── fsm/                       ← 有限状态机约束解码
```

---

### 3.3 关键模块精讲

#### RadixCache 前缀匹配流程

```python
# radix_cache.py（简化）
def match_prefix(self, token_ids):
    """返回最长匹配前缀对应的 KV Cache blocks"""
    node = self.root
    matched_len = 0
    for token in token_ids:
        child = node.children.get(token)
        if child is None:
            break
        node = child
        matched_len += len(node.key)   # key 是压缩后的 token 序列
    return matched_len, node.value     # value 是物理 block 索引列表
```

#### CUDAGraph 优化

- decode 阶段 batch size 固定（通常为 1~几十），适合 CUDAGraph capture
- 首次运行时 capture 计算图，后续直接 replay，kernel launch 开销接近零
- SGLang 对不同 batch size 分别 capture，运行时选择最接近的图

---

## 4. vLLM vs SGLang 横向对比

| 维度 | vLLM | SGLang |
|------|------|--------|
| KV Cache 管理 | PagedAttention（块级哈希 APC） | RadixAttention（前缀树，粒度更细） |
| 缓存命中率 | 较高 | 更高（3~5× vs vLLM APC） |
| 进程架构 | 单进程 AsyncEngine + Worker | 多进程（ZMQ IPC），调度与推理解耦 |
| 结构化输出 | 通过 outlines 集成 | 原生支持（Jump-Forward Decoding） |
| 推理延迟（TTFT） | 优秀 | 更优（CUDAGraph + RadixAttention） |
| 生态成熟度 | 更成熟，模型支持更广 | 较新，快速追赶 |
| 代码复杂度 | 较高 | 稍低，更易二次开发 |
| 适合场景 | 通用生产部署 | 高重复前缀、结构化输出场景 |

---

## 5. 动手实验清单

按顺序完成，每项预计 0.5~2 天：

- [ ] **实验1**：跑通 vLLM 离线推理，用 `--enable-prefix-caching` 前后对比 TTFT
- [ ] **实验2**：在 vLLM 源码中加日志，打印每个请求的 block 分配/释放过程
- [ ] **实验3**：实现一个最简化版 PagedAttention（纯 Python，不要 GPU），加深理解
- [ ] **实验4**：跑通 SGLang，对比相同 workload 下 vLLM 和 SGLang 的吞吐与延迟
- [ ] **实验5**：在 SGLang 中构造高前缀重叠的请求，观察 RadixCache 命中率变化
- [ ] **实验6**：阅读 `radix_cache.py`，手写一个 RadixTree 的单元测试，覆盖 insert / match / evict
- [ ] **实验7**：修改 vLLM scheduler，实现一个简单的优先级调度策略
- [ ] **实验8**：在 SGLang 中使用结构化输出（JSON Schema），理解约束解码流程
- [ ] **实验9**：对 vLLM 或 SGLang 跑 profiling（`nsys` 或 `torch.profiler`），找出推理瓶颈

---

## 6. 推荐资料

### 论文

| 论文 | 内容 |
|------|------|
| [vLLM: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | vLLM 原始论文 |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | SGLang 原始论文 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 理解 attention 计算优化基础 |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Continuous Batching 鼻祖 |

### 源码 & 文档

- vLLM 官方文档：https://docs.vllm.ai
- SGLang 官方文档：https://sglang.readthedocs.io
- vLLM GitHub：https://github.com/vllm-project/vllm
- SGLang GitHub：https://github.com/sgl-project/sglang
- DeepWiki SGLang 架构图：https://deepwiki.com/sgl-project/sglang

### 博客 & 讲解

- [vLLM 内核探秘（掘金系列）](https://juejin.cn/post/7628881020147154980) — 中文，逐模块讲解
- [手撕 SGLang KV Cache 核心逻辑（知乎）](https://zhuanlan.zhihu.com/p/1994495318197305400) — RadixAttention 源码精讲
- [vLLM 高吞吐推理系统全景拆解（CSDN）](https://suanli.blog.csdn.net/article/details/159724638) — 系统性中文讲解

### 学习建议

1. **先跑起来，再看源码**：先用 API 感受两个框架的行为差异，再带着问题看代码
2. **配合 debug 看代码**：用 `pdb` 或 VSCode debugger 在关键函数上打断点，跟踪真实请求的执行路径
3. **画图辅助理解**：KV Cache 的 block 分配、RadixTree 的节点结构，都值得手画数据结构图
4. **对比阅读**：同一个问题（如 KV Cache 管理）在两个框架中的不同实现对比阅读，理解更深
5. **提交 PR**：找一个小 bug 或文档改进提 PR，是最快融入社区的方式
