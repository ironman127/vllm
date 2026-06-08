# vLLM v1 架构文档

> 本文档基于 vLLM v1 源码梳理，覆盖整体分层架构、各组件职责、并行模式以及组件间分工协作关系。

---

## 1. 整体分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                      客户端 / HTTP 请求                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/OpenAI API
┌───────────────────────────▼─────────────────────────────────┐
│                      API Server                              │
│   FastAPI + AsyncLLM                                         │
│   - 接收 HTTP 请求，转换为 Request 对象                        │
│   - Detokenizer：将 token_id 还原为文字                        │
│   - OutputProcessor：整理 logprobs、stop 条件                  │
│   - 通过 EngineCoreClient (ZMQ) 与 EngineCore 通信            │
└───────────────────────────┬─────────────────────────────────┘
                            │ ZMQ DEALER/PUSH socket
                            │ (msgspec 序列化)
┌───────────────────────────▼─────────────────────────────────┐
│                  EngineCoreProc（独立子进程）                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    EngineCore                        │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │              Scheduler                       │   │    │
│  │  │  - waiting / running / skipped_waiting 队列  │   │    │
│  │  │  - 连续批处理 / 前缀缓存 / chunked prefill   │   │    │
│  │  │  - 投机解码 / LoRA / 多模态 budget           │   │    │
│  │  │  - 输出：SchedulerOutput                     │   │    │
│  │  └────────────────────┬────────────────────────┘   │    │
│  │                        │                             │    │
│  │  ┌─────────────────────▼──────────────────────┐    │    │
│  │  │           KVCacheManager (门面层)            │    │    │
│  │  │  KVCacheCoordinator                          │    │    │
│  │  │  ├── UnifiedKVCacheCoordinator（共用池）     │    │    │
│  │  │  └── SeparatedKVCacheCoordinator（分离池）   │    │    │
│  │  │       └── BlockPool (物理块池 + LRU)         │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                                                      │    │
│  └────────────────────────┬─────────────────────────────┘   │
│                            │ execute_model(SchedulerOutput)   │
│  ┌─────────────────────────▼──────────────────────────────┐  │
│  │                     Executor                            │  │
│  │  - 知道 TP/PP/EP 拓扑（Worker 进程数、rank 分配）       │  │
│  │  - 将 SchedulerOutput 广播给所有 Worker                  │  │
│  │  - 收集所有 Worker 返回的 ModelRunnerOutput              │  │
│  └─────────────────────────┬──────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────-┘
                             │ MessageQueue（SHM+ZMQ IPC 或 ZMQ TCP）+ NCCL
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
┌─────▼──────┐        ┌──────▼─────┐        ┌──────▼──────┐
│  Worker 0  │◀──────▶│  Worker 1  │◀──────▶│  Worker N   │
│  (GPU 0)   │ NCCL   │  (GPU 1)   │ NCCL   │  (GPU N)    │
│            │        │            │        │             │
│ ModelRunner│        │ ModelRunner│        │ ModelRunner │
│ CacheEngine│        │ CacheEngine│        │ CacheEngine │
└────────────┘        └────────────┘        └─────────────┘
```

---

## 2. 各层职责

### 2.1 API Server（`v1/engine/async_llm.py`）

| 职责 | 说明 |
|------|------|
| 请求接收 | HTTP → `Request` 对象，分配 `request_id` |
| 输入预处理 | `InputProcessor`：tokenize、多模态 hash |
| 异步通信 | 通过 `EngineCoreClient` 向 EngineCore 发送 ADD/ABORT 消息 |
| 输出处理 | `Detokenizer`：token_id → 文字；`OutputProcessor`：logprobs、stop |
| 流式响应 | AsyncGenerator 逐步把结果 yield 给客户端 |

### 2.2 EngineCoreProc（`v1/engine/core.py`）

独立子进程，封装 EngineCore，提供进程边界隔离：

- **启动握手**：子进程启动后绑定 ZMQ socket，加载权重、分配 KV cache 完成后向前端发送 `EngineCoreReadyResponse`
- **主循环**（busy-loop）：
  ```
  while True:
      轮询 input_queue（ADD/ABORT/UTILITY 消息）
      engine.step_fn()
          → scheduler.schedule()
          → executor.execute_model()
          → scheduler.update_from_output()
      将 EngineCoreOutputs 通过 output_queue 发回前端
  ```
- **DP 模式**：`DPEngineCoreProc` 子类通过 NCCL all-reduce 同步多个 EngineCore 实例的负载状态

### 2.3 Scheduler（`v1/core/sched/scheduler.py`）

**感知不到任何硬件并行拓扑**，只处理逻辑层面的请求调度：

#### 请求状态机

```
WAITING ──────────────────────────────────▶ RUNNING
  │  (KV block 分配成功 / 前缀命中)            │
  │                                           │ (output token 生成)
  ◀─────────── PREEMPTED ◀────────────────────┤
  │            (显存不足，抢占)                │
  ▼                                           │
FINISHED ◀──────────────────────────────────┘
  (EOS / 长度上限 / 客户端 abort)
```

#### 三个核心队列

| 队列 | 含义 |
|------|------|
| `waiting` | 等待首次调度的请求（FCFS 或 Priority） |
| `running` | 当前步正在执行的请求 |
| `skipped_waiting` | 本轮 waiting 中因资源不足被跳过的请求 |

#### `schedule()` 主流程

```
1. 处理 running 请求（decode/speculative decode）
2. 从 waiting 队列取请求，尝试分配 KV blocks
   - 前缀命中 → touch cached blocks
   - 新分配 → get_new_blocks()
3. 构造 SchedulerOutput（NewRequestData + CachedRequestData）
4. 若资源不足 → 抢占最低优先级的 running 请求
```

### 2.4 KV Cache 管理体系

```
KVCacheManager（门面，调度器视角）
    └── KVCacheCoordinator（拓扑变体）
         ├── UnifiedKVCacheCoordinator  → 所有 attention group 共用一个 BlockPool
         └── SeparatedKVCacheCoordinator → 各 attention group 独立 BlockPool
              └── BlockPool（物理块池）
                   ├── FreeKVCacheBlockQueue（双向链表，LRU 顺序）
                   └── BlockHashToBlockMap（hash → block，前缀缓存索引）
```

**关键数据结构**：

```python
KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # 外层 tuple：固定长度 = KV cache group 数（不同 attention 类型）
    # 内层 Sequence：可变长度 = 该 group 分配的物理块列表
```

**LRU 淘汰机制**：  
`ref_cnt == 0` 的块进入 `FreeKVCacheBlockQueue` 尾部（has_content=True），再次分配时若 free list 不足，从头部淘汰最久未访问的块并从哈希索引中移除。

### 2.5 Executor（`v1/worker/`）

**运行时不做任何调度决策**，只负责分发：

```python
# 伪代码
def execute_model(scheduler_output):
    # 将同一份 SchedulerOutput 广播给所有 Worker
    for worker in all_workers:
        worker.execute_model(scheduler_output)
    # 汇聚结果
    return collect_outputs()
```

Executor 在**初始化阶段**知道拓扑（TP/PP size、rank 数量），但运行时不区分各 Worker 的具体工作。每个 Worker 在加载权重时就已确定自己负责哪部分（权重分片或哪些层）。

### 2.6 Worker（`v1/worker/gpu_worker.py`）

每个 Worker 持有：
- **ModelRunner**：负责实际的模型 forward pass
- **CacheEngine**：管理本 GPU 上的 KV cache 物理内存（`torch.Tensor`）

Worker 自行协调节点间数据依赖，Executor 完全不介入：

---

## 3. 并行模式

### 3.1 Tensor Parallelism（TP）

```
同一层权重按列/行切分到 N 个 GPU
每个 Worker 执行自己的分片，forward 过程中自行发起 NCCL all_reduce

Executor:  broadcast(SchedulerOutput) ──▶ Worker 0, 1, ..., N-1
Worker 0:  W[:, 0:k] × x  ─┐
Worker 1:  W[:, k:2k] × x  ├──▶ NCCL all_reduce ──▶ 完整输出
Worker N:  W[:, (N-1)k:Nk]×x ─┘
```

- Executor 对所有 Worker 广播**相同**指令
- all_reduce 通信由模型算子（`RowParallelLinear` 等）内部发起
- 延迟敏感：每个 forward step 都有同步点

### 3.2 Pipeline Parallelism（PP）

```
不同 PP stage 的 Worker 负责不同的层

Stage 0 (层 0~L/2):   Worker 0,1  ──激活值──▶  Stage 1 (层 L/2~L): Worker 2,3
                                      P2P send/recv
```

- 每个 Worker 初始化时只加载本 stage 的层权重
- 激活值通过 P2P `send/recv` 传递，由 Worker 自行发起
- KV cache 分布在各 stage 节点，每个节点只缓存本 stage 层的 KV

### 3.3 Expert Parallelism（EP，MoE 模型）

```
不同 Expert 权重分布在不同 GPU
token 根据 router 打分做 all_to_all 路由到对应 Expert 节点计算

NCCL all_to_all：由 MoE 层 router 代码内部发起
```

### 3.4 Data Parallelism（DP）

```
多个独立的 EngineCore 实例（每个对应独立的 GPU 组）
由 DPCoordinator（coordinator.py）做请求级负载均衡

API Server ──▶ DPCoordinator ──┬──▶ EngineCore 0 (GPU 组 0)
                               ├──▶ EngineCore 1 (GPU 组 1)
                               └──▶ EngineCore N (GPU 组 N)
```

- 各 EngineCore **完全独立**，不共享 KV cache
- DP 与 TP/PP 可叠加：每个 DP 副本内部再做 TP=8 跨节点

### 3.5 各并行模式对比

| 维度 | TP | PP | EP | DP |
|------|----|----|----|----|
| 切分粒度 | 权重矩阵的行/列 | 模型层 | MoE Expert | 请求 |
| 通信类型 | all_reduce（每 step）| P2P send/recv（层间）| all_to_all（MoE 层）| 无（请求级隔离）|
| 谁发起通信 | Worker（算子内部） | Worker（stage 边界）| Worker（router）| 无 |
| Executor 感知 | 初始化时知道 TP size | 初始化时知道 PP size | 初始化时知道 EP size | N/A（各自独立）|
| 延迟影响 | 高（每 step 同步）| 中（bubble 问题）| 中（all_to_all）| 无 |

---

## 4. P/D 分离（Prefill-Decode Disaggregation）

### 4.1 动机

| 阶段 | 瓶颈 | 批处理特性 |
|------|------|-----------|
| Prefill | **计算密集**（大矩阵乘法） | 一次处理所有输入 token |
| Decode  | **内存带宽密集**（每步只生成 1 token）| 连续小批量 |

两者混合部署时，Prefill 的大计算量会打断 Decode 的连续生成，造成延迟抖动。

### 4.2 架构

```
客户端
  │
  ▼
API Server ──▶ Decode 集群（请求所有权在此）
                  │
                  │ 1. Decode EngineCore 创建请求，通知 P 集群做 prefill
                  ▼
              Prefill 集群（"外包" prefill 计算）
                  │
                  │ 2. Prefill 完成后，通过 KVConnector (RDMA/TCP) 将 KV 传回
                  ▼
              Decode 集群
                  │ 3. Decode EngineCore 收到 KV，开始 decode 推理
```

### 4.3 KVConnector 双角色

```python
class KVConnector:
    # Role=SCHEDULER（运行在 EngineCore 进程）
    #   get_num_new_matched_tokens()   ← 告知调度器有多少 token 的 KV 已就绪
    #   update_state_after_alloc()     ← 分配完 KV block 后更新连接状态
    
    # Role=WORKER（运行在 Worker 进程）
    #   start_load_kv()                ← 触发从远端拉取 KV（RDMA/TCP）
    #   save_kv_layer()                ← 将本层 KV 写入传输缓冲
```

- `num_external_computed_tokens`：KVConnector 管理的 token 数量，对应的 KV 不占用本地 BlockPool

### 4.4 与 TP/PP 的本质区别

| 架构 | 分工维度 | KV 传输 | 请求所有权 |
|------|---------|---------|-----------|
| P/D 分离 | 按**推理阶段**（prefill vs decode）| 跨集群传整个 KV | Decode 集群 |
| TP | 按**权重维度**（同一层内） | 无（各持权重分片）| 共同执行 |
| PP | 按**模型层** | 传激活值（非 KV）| 共同执行 |

> 实际部署可叠加：P 集群内部 TP=8，D 集群内部 TP=8，两集群之间再做 P/D 分离。

---

## 5. 组件间通信总结

```
┌─────────────┐   ZMQ DEALER/PUSH     ┌──────────────────┐
│  API Server │ ─────────────────────▶│  EngineCoreProc  │
│  (前端进程) │ ◀───────────────────── │  (后端子进程)    │
└─────────────┘   msgspec 序列化        └────────┬─────────┘
                                                  │
                                    MessageQueue
                                    （同节点：SHM ring buffer + ZMQ IPC
                                      跨节点：ZMQ TCP XPUB/SUB）
                                                  │
                              ┌───────────────────┼───────────────────┐
                              │                   │                   │
                         ┌────▼────┐        ┌─────▼───┐        ┌─────▼───┐
                         │Worker 0 │        │Worker 1 │        │Worker N │
                         └────┬────┘        └─────┬───┘        └─────┬───┘
                              │                   │                   │
                              └───────────────────┴───────────────────┘
                                          NCCL（TP all_reduce）
                                          P2P send/recv（PP 激活值）
                                          all_to_all（EP MoE 路由）
```

| 通信链路 | 协议 | 方向 | 内容 |
|---------|------|------|------|
| API Server → EngineCore | ZMQ + msgspec | 双向 | Request / EngineCoreOutputs |
| Executor → Worker（同节点）| MessageQueue：SHM ring buffer + ZMQ IPC 通知 | 单向广播 | RPC 调用（含 SchedulerOutput）|
| Executor → Worker（跨节点）| MessageQueue：ZMQ TCP XPUB/SUB | 单向广播 | RPC 调用（含 SchedulerOutput）|
| Worker → Executor（输出汇聚）| MessageQueue（同/跨节点对称，仅 output_rank 写入）| 单向 | ModelRunnerOutput |
| Worker ↔ Worker（TP）| AllReduce 多后端：NCCL SymmMem / Custom AR（同节点 NVLink）/ PyNCCL（跨节点 IB/RoCE）| 双向 | 激活值分片 |
| Worker ↔ Worker（PP）| NCCL P2P send/recv | 单向（流水）| 激活值 |
| Prefill Worker → Decode Worker | KVConnector（RDMA/TCP）| 单向 | KV cache 数据 |

### 5.1 MessageQueue 双模式架构（Executor ↔ Worker 控制通道）

`MessageQueue`（`vllm/distributed/device_communicators/shm_broadcast.py`）同时支持本地和远程两类读者，由可序列化的 `Handle` 在进程间传递：

```
写端（Executor / EngineCoreProc）
  ├─── 本地读者（同节点 Worker）
  │     写入：ShmRingBuffer（multiprocessing.shared_memory 无锁环形缓冲）
  │     通知：SpinCondition via ZMQ IPC ipc:// socket
  │           （busy-loop 1s → 降级为 ZMQ poll 1ms/次）
  │
  └─── 远程读者（跨节点 Worker，多节点 PP / DP 场景）
        传输：ZMQ TCP  tcp://ip:port  XPUB/SUB 模式（含完整序列化数据）
```

| 组件 | 说明 |
|------|------|
| `ShmRingBuffer` | flag 数组 `[written_flag \| r0_flag \| … \| rN_flag]`；`memory_fence()`（threading.Lock）保证跨进程内存可见性 |
| `SpinCondition` | 写端通过 ZMQ IPC `notify()`；读端先 busy-loop 1 s，再降级为 1 ms poll |
| `Handle` | 序列化描述符：`buffer_handle`（SHM 段名）、`local_subscribe_addr`（IPC 路径）、`remote_subscribe_addr`（TCP 地址）|

**输出汇聚路径**：每个 Worker 有独立的单读者 `response_mq`；仅 `output_rank`（= last PP stage 的首个 TP rank = `world_size − tp_size`）写入 `ModelRunnerOutput`，Executor 从 `response_mqs[output_rank]` 读取。

### 5.2 TP AllReduce 多后端调度

`CudaCommunicator.all_reduce()`（`vllm/distributed/device_communicators/cuda_communicator.py`）按以下优先级选择后端：

| 优先级 | 后端 | 适用条件 |
|--------|------|---------|
| 1 | NCCL SymmMem | NCCL 版本支持对称内存 |
| 2 | QuickReduce | ROCm 平台 |
| 3 | FlashInfer AR | FlashInfer 可用 |
| 4 | Custom AllReduce | **同节点** NVLink；world_size ∈ {2, 4, 6, 8} |
| 5 | SymmMem | 对称内存可用 |
| 6 | PyNCCL | 通用回退（含跨节点 InfiniBand / RoCE）|

> 跨节点 TP 时，`in_the_same_node_as()` 集合检测失败 → Custom AllReduce 自动禁用 → 回退到 PyNCCL over InfiniBand/RoCE。

---

## 6. 一次完整请求的生命周期

```
1. 客户端发送 HTTP 请求
        │
2. API Server：tokenize → 创建 Request → ZMQ 发送 ADD_REQUEST
        │
3. EngineCore：加入 waiting 队列
        │
4. Scheduler.schedule()：
   a. 尝试前缀缓存命中（BlockPool hash 查找）
   b. 分配 KV blocks（BlockPool.get_new_blocks()）
   c. 生成 SchedulerOutput（NewRequestData）
        │
5. Executor.execute_model(SchedulerOutput)：
   a. 广播给所有 Worker
   b. 各 Worker 并行执行 ModelRunner.forward()
   c. Worker 间 NCCL 通信（TP all_reduce 等）
   d. 返回 token_ids（ModelRunnerOutput）
        │
6. Scheduler.update_from_output()：
   a. 追加新 token，更新请求状态
   b. 检查停止条件（EOS / max_len / stop_string）
   c. cache_full_blocks()：将完整块写入前缀缓存索引
        │
7. EngineCore → ZMQ → API Server：发送 EngineCoreOutputs
        │
8. API Server：Detokenizer → 流式响应给客户端
        │
9. 请求结束：free_blocks() → 归还 KV blocks → 更新 LRU 队列
```

---

## 7. 核心设计原则

1. **分层隔离**：Scheduler 感知不到硬件拓扑；Executor 感知拓扑但不介入通信；Worker 自行协调并行通信
2. **门面模式**：`KVCacheManager` 对 Scheduler 暴露统一 API，屏蔽 `UnifiedCoordinator` vs `SeparatedCoordinator` 的拓扑差异
3. **进程隔离**：API Server 和 EngineCore 跑在不同进程，通过 ZMQ 通信，GPU OOM 不会影响前端
4. **LRU 前缀缓存**：物理块双向链表 + 哈希索引，O(1) 命中查找和淘汰，显著降低重复请求的 KV 计算
5. **采样指标**：`KVCacheMetricsCollector` 以 1% 采样率跟踪块生命周期，不影响热路径性能
