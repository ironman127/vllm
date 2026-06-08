# vLLM Tensor Parallel Worker Communication and Synchronization

## Overview
vLLM implements sophisticated communication and synchronization mechanisms for Tensor Parallel (TP) execution across multiple GPUs. The system uses a hierarchical architecture combining PyTorch distributed communication, custom optimizations, and shared memory broadcasting.

## 1. Worker Initialization and Process Group Setup for TP

### 1.1 Distributed Environment Initialization
**File**: `vllm/distributed/parallel_state.py`

The initialization follows a two-stage process:

1. **Global Distributed Environment** (`init_distributed_environment`):
   - Initializes PyTorch's global process group with NCCL backend
   - Sets up world group with all ranks
   - Handles data parallelism adjustments across nodes
   - Supports elastic EP (Elastic Pipeline) configurations

2. **Model Parallel Initialization** (`initialize_model_parallel`):
   - Creates TP (Tensor Parallel) groups from global ranks
   - Creates separate process groups for:
     - TP (Tensor Parallel)
     - PP (Pipeline Parallel)
     - DP (Data Parallel)
     - EP (Expert Parallel)
     - DCP (Decode Context Parallel)
     - PCP (Prefill Context Parallel)

**Layout**: The rank layout is organized as:
```
ExternalDP × DP × PP × TP
```

For TP specifically, the process group is constructed such that:
- Workers with the same TP rank are grouped together
- Each TP group is created from world ranks that are sequential after reshaping

**Example**: With world_size=8, tensor_parallel_size=2, pipeline_parallel_size=2, data_parallel_size=1:
```
TP Groups:
  [0, 1], [2, 3], [4, 5], [6, 7]
```

### 1.2 GroupCoordinator Architecture
**File**: `vllm/distributed/parallel_state.py` (lines 290-1142)

The `GroupCoordinator` class manages all communication within a group:

```python
class GroupCoordinator:
    rank: int                           # Global rank
    ranks: list[int]                    # Global ranks in group
    world_size: int                     # Group size
    rank_in_group: int                  # Rank within group
    local_rank: int                     # Local device index
    
    cpu_group: ProcessGroup             # Gloo backend (CPU communication)
    device_group: ProcessGroup          # NCCL/XPU backend (GPU communication)
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: MessageQueue | None # Shared memory broadcaster
    use_custom_op_call: bool            # Use custom ops for collectives
```

**Key attributes**:
- **Device**: Automatically set based on platform (cuda:local_rank, xpu:local_rank, etc.)
- **Device Communicator**: Lazy initialized if `use_device_communicator=True` and world_size > 1
- **Message Queue Broadcaster**: Optional high-performance shared memory broadcast for TP groups

### 1.3 TP Group Creation
**Code snippet from `initialize_model_parallel`**:

```python
# Build the tensor model-parallel groups.
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
group_ranks = [x.tolist() for x in group_ranks]
_TP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_message_queue_broadcaster=True,  # Only TP uses MQ broadcaster
    group_name="tp",
)
```

The TP group is special in that:
- It's the only group that uses message queue broadcaster by default
- Device communicator is enabled for optimized collectives
- Custom all-reduce can be used within TP

## 2. Input Broadcast Mechanisms

### 2.1 Scheduler Output Broadcasting
**Files**: 
- `vllm/v1/executor/multiproc_executor.py` (lines 150-156)
- `vllm/distributed/device_communicators/shm_broadcast.py`

The scheduler output (batch data) is broadcast from the driver worker to all workers using shared memory:

```python
# In MultiprocExecutor._init_executor()
self.rpc_broadcast_mq = MessageQueue(
    self.world_size,
    self.local_world_size,
    max_chunk_bytes=max_chunk_bytes,
    connect_ip=mq_connect_ip,
)
scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
```

### 2.2 MessageQueue (SharedMemory Broadcast)
**File**: `vllm/distributed/device_communicators/shm_broadcast.py`

MessageQueue provides ultra-low-latency broadcast using:

1. **Producer** (Rank 0):
   - Writes scheduler output to shared memory ring buffer
   - Sends notifications over ZMQ (PUB socket)

2. **Consumers** (All other ranks):
   - Subscribe to ZMQ notifications
   - Read from shared memory ring buffer with proper memory fencing
   - Spin-wait for high throughput, fall back to idle wait if no activity

**Memory synchronization**:
```python
def memory_fence():
    """Full memory barrier for shared memory synchronization."""
    with _memory_fence_lock:
        pass
```

**Features**:
- Lock-free ring buffer for high throughput
- Adaptive busy-waiting: spins for active periods, sleeps during idle
- ZMQ CONFLATE mode prevents message queue buildup under high load
- Supports zero-copy pickling with `PickleBuffer`

**SpinCondition** (lines 97-150):
- Readers spin-loop on shared memory for configurable `busy_loop_s` (default 1s)
- Falls back to polling ZMQ socket after busy period
- Separate process-local socket for cancellation signaling

### 2.3 Tensor Dictionary Broadcasting
**File**: `vllm/distributed/parallel_state.py` (lines 737-817)

The `broadcast_tensor_dict` method optimizes tensor transmission:

1. **Split Phase**: Separates metadata from tensors
   - Tensors → metadata list with device/dtype/size
   - Actual tensor list

2. **Metadata Broadcast**: Via CPU group (Gloo)
3. **Tensor Broadcast**: Async via device group (NCCL)

```python
def broadcast_tensor_dict(self, tensor_dict, src=0):
    metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
    # Broadcast metadata via CPU group
    self.broadcast_object(metadata_list, src=src)
    # Async broadcast tensors via device group
    async_handles = []
    for tensor in tensor_list:
        handle = torch.distributed.broadcast(
            tensor, src=self.ranks[src], 
            group=self.device_group, 
            async_op=True
        )
        async_handles.append(handle)
    for handle in async_handles:
        handle.wait()
```

## 3. AllReduce and Collective Operations During Forward Pass

### 3.1 AllReduce Flow
**File**: `vllm/distributed/parallel_state.py` (lines 514-542)

```python
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    if self.world_size == 1:
        return input_
    
    if self.use_custom_op_call:
        return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
    else:
        return self._all_reduce_out_place(input_)

def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
    if self.device_communicator is None:
        raise ValueError("No device communicator found")
    return self.device_communicator.all_reduce(input_)
```

**Custom Op Path** (TPU/CPU): Uses PyTorch's functional collectives registered as custom ops
**Device Communicator Path** (GPU): Dispatches to device-specific implementation

### 3.2 AllReduce Backend Selection
**File**: `vllm/distributed/device_communicators/cuda_communicator.py` (lines 241-260)

vLLM supports multiple all-reduce backends with automatic dispatch:

```python
def all_reduce(self, input_):
    # Dispatch order based on input characteristics
    
    1. NCCL_SYMM_MEM (PyTorch Symmetric Memory + NCCL)
       - Requirements: Symmetric memory enabled, batch invariant off, specific world sizes
       - Optimization: Zero-copy communication via symmetric memory allocator
    
    2. QUICK_REDUCE (AMD-specific)
       - Only on ROCm MI300 series
       - Custom implementation optimized for AMD hardware
    
    3. FLASHINFER
       - Third-party optimized all-reduce
       - Size/dtype gates for applicability
    
    4. CUSTOM_ALLREDUCE
       - vLLM's custom implementation
       - NVLink required (same-node only)
       - Supports world sizes: [2, 4, 6, 8]
    
    5. SYMM_MEM
       - PyTorch's symmetric memory backend
       - CPU-free, GPU-to-GPU
    
    6. PYNCCL (Fallback)
       - Direct NCCL wrapper
       - Used as final fallback
```

### 3.3 Custom AllReduce Implementation
**File**: `vllm/distributed/device_communicators/custom_all_reduce.py`

Provides optimized all-reduce for same-node TP groups:

```python
class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    
    def __init__(self, group, device, max_size=8192*1024, symm_mem_enabled=False):
        # Validation
        - Must be single-node (in_the_same_node_as check)
        - World size must be supported
        - NVLink must be available (P2P access check)
        
        # Initialization
        - Gather physical device IDs from all ranks
        - Test NVLink connectivity
        - Set up custom kernel infrastructure
```

**Optimization**: 
- Uses GPU peer-to-peer access via NVLink
- Bypasses main memory for intra-node communication
- Custom kernels for high throughput

### 3.4 PyNCCL Communicator
**File**: `vllm/distributed/device_communicators/pynccl.py`

Direct Python wrapper around NCCL library:

```python
class PyNcclCommunicator:
    def __init__(self, group, device, library_path=None):
        # Get unique ID from rank 0
        self.unique_id = nccl.ncclGetUniqueId() if rank == 0 else empty_id
        
        # Broadcast unique ID to all ranks
        dist.broadcast(tensor_from_id, src=ranks[0], group=group)
        
        # Initialize NCCL communicator
        with torch.accelerator.device_index(device.index):
            self.comm = nccl.ncclCommInitRank(world_size, unique_id, rank)
            
            # Warmup all-reduce
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            data.synchronize()
```

### 3.5 AllGather and ReduceScatter
**File**: `vllm/distributed/parallel_state.py`

```python
def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if self.world_size == 1:
        return input_
    if self.use_custom_op_call:
        return torch.ops.vllm.all_gather(
            input_, dim, self.world_size, group_name=self.unique_name
        )
    else:
        return self._all_gather_out_place(input_, dim)

def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if self.world_size == 1:
        return input_
    if self.use_custom_op_call:
        return torch.ops.vllm.reduce_scatter(
            input_, dim, self.world_size, group_name=self.unique_name
        )
    else:
        return self._reduce_scatter_out_place(input_, dim)
```

**Device Communicator handles**:
- `all_gather`: Concatenates tensors along dimension
- `reduce_scatter`: Splits and reduces tensors along dimension
- Both support variable-sized inputs (`all_gatherv`, `reduce_scatterv`)

## 4. Worker-to-Engine Communication

### 4.1 Executor-Worker RPC Pattern
**File**: `vllm/v1/executor/multiproc_executor.py` (lines 306-348)

```python
def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False):
    return self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,  # Get result from one rank only
        non_block=non_block,
        timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        kv_output_aggregator=self.kv_output_aggregator,
    )
```

**Key aspects**:
- **Collective RPC**: All workers participate, but only one returns result
- **Output Rank**: Usually rank 0 or a driver worker
- **KV Output Aggregator**: Optional aggregation of KV caches from multiple ranks
- **Non-blocking**: Returns Future for async execution

### 4.2 Message Queue for Results
**File**: `vllm/v1/executor/multiproc_executor.py` (lines 206-219)

Each worker has a response message queue:

```python
self.response_mqs = []
for rank in range(self.world_size):
    if rank < self.local_world_size:
        local_mq = self.workers[rank].worker_response_mq
        self.response_mqs.append(local_mq)
    else:
        # Remote workers' response MQs via peer channels
        remote_mq = self.workers[0].peer_worker_response_mqs[rank]
        self.response_mqs.append(remote_mq)
```

### 4.3 Future Handling
**File**: `vllm/v1/executor/multiproc_executor.py` (lines 69-100)

```python
class FutureWrapper(Future):
    def __init__(self, futures_queue, get_response, aggregate=lambda x: x):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
    
    def result(self, timeout=None):
        # Drain futures ahead in queue
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()
    
    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            self.set_result(response)
        except Exception as e:
            self.set_exception(e)
```

**Features**:
- Queued futures maintain order
- Automatic aggregation of responses
- Proper exception propagation

## 5. Custom Communication Libraries

### 5.1 All-Reduce Utils
**File**: `vllm/distributed/device_communicators/all_reduce_utils.py`

Provides configuration and helpers:
- `should_nccl_symm_mem_allreduce()`: Predicate for symmetric memory dispatch
- `gpu_p2p_access_check()`: Validates P2P connectivity
- `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`: Tuning parameters per world size

### 5.2 Custom AllReduce Strategies
**Files**:
- `custom_all_reduce.py`: NVLink-based peer-to-peer
- `quick_all_reduce.py`: AMD MI300 specific
- `flashinfer_all_reduce.py`: Third-party integration
- `symm_mem.py`: PyTorch symmetric memory wrapper
- `pynccl_allocator.py`: NCCL memory management

### 5.3 All-to-All Communication (MoE)
**File**: `vllm/distributed/device_communicators/all2all.py`

Multiple implementations for expert parallel:
- `AgRsAll2AllManager`: AllGather + ReduceScatter
- `DeepEPHTAll2AllManager`: High-throughput DeepEP
- `DeepEPLLAll2AllManager`: Low-latency DeepEP
- `MoriAll2AllManager`: Custom implementation
- `NixlEPAll2AllManager`: NixL integration
- `FlashInferNVLinkTwoSidedManager`: FlashInfer NVLink
- `FlashInferNVLinkOneSidedManager`: One-sided variant

## 6. Executor/Worker Architecture for TP

### 6.1 Executor Hierarchy
**File**: `vllm/v1/executor/abstract.py`

```python
class Executor(ABC):
    uses_ray: bool = False
    supports_pp: bool = False
    
    @staticmethod
    def get_class(vllm_config) -> type["Executor"]:
        # Selection based on distributed_executor_backend
        backends:
        - "ray" → RayDistributedExecutor / RayExecutorV2
        - "mp" → MultiprocExecutor
        - "uni" → UniProcExecutor
        - "external_launcher" → ExecutorWithExternalLauncher
```

### 6.2 MultiprocessExecutor (Default for TP)
**File**: `vllm/v1/executor/multiproc_executor.py`

Manages TP via multiprocessing:

```python
class MultiprocExecutor(Executor):
    supports_pp: bool = True
    
    def _init_executor(self):
        # 1. Setup distributed environment
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        
        # 2. Create message queues
        self.rpc_broadcast_mq = MessageQueue(...)  # Leader node
        
        # 3. Create worker processes
        for local_rank in range(self.local_world_size):
            unready_worker_handle = WorkerProc.make_worker_process(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=global_rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
            )
        
        # 4. Wait for workers to initialize
        self.workers = WorkerProc.wait_for_ready(unready_workers)
```

### 6.3 RayExecutor (Multi-node TP)
**File**: `vllm/v1/executor/ray_executor.py`

Ray-based distributed execution:

```python
class RayDistributedExecutor(Executor):
    uses_ray: bool = True
    supports_pp: bool = True
    
    def _init_executor(self):
        # 1. Initialize Ray cluster
        initialize_ray_cluster(self.parallel_config)
        
        # 2. Create Ray remote workers
        self._init_workers_ray(placement_group)
        
        # 3. Setup KV connector for KV cache transfer
        self.has_connector = self.vllm_config.kv_transfer_config is not None
```

### 6.4 Worker Process Lifecycle
**File**: `vllm/v1/executor/multiproc_executor.py`

```python
class WorkerProc:
    @staticmethod
    def make_worker_process(...):
        # Create worker process with lazy initialization
        proc = context.Process(
            target=WorkerProc.worker_main,
            args=(vllm_config, local_rank, rank, ...),
            name=f"WorkerProcess-{rank}",
        )
        proc.start()
        return UnreadyWorkerProcHandle(proc, death_pipe, ready_pipe)
    
    @staticmethod
    def worker_main(...):
        # 1. Update environment variables
        # 2. Initialize distributed environment
        init_distributed_environment(...)
        ensure_model_parallel_initialized(...)
        
        # 3. Create worker instance
        worker = worker_wrapper.init_worker(...)
        
        # 4. Signal ready
        ready_pipe.send(True)
        
        # 5. Main loop: receive scheduler outputs, execute model
        while True:
            scheduler_output = rpc_broadcast_mq.recv()
            output = worker.execute_model(scheduler_output)
            worker_response_mq.send(output)
```

### 6.5 Driver vs Non-Driver Workers
**File**: `vllm/v1/executor/multiproc_executor.py` (lines 264-265)

```python
def _is_driver_worker(self, rank: int) -> bool:
    return rank % self.parallel_config.tensor_parallel_size == 0
```

**Driver workers** (TP rank 0 in each DP group):
- Receive scheduler outputs first
- Broadcast to non-driver workers
- Return outputs to executor

**Non-driver workers** (TP rank > 0):
- Participate in TP collectives
- No external communication

## 7. Synchronization Mechanisms

### 7.1 Barrier Synchronization
**File**: `vllm/distributed/parallel_state.py` (lines 1052-1059)

```python
def barrier(self):
    """Barrier synchronization among the group."""
    torch.distributed.barrier(group=self.cpu_group)
```

**Important**: Uses CPU group (Gloo) not device group (NCCL)
- Reason: NCCL barrier is unreliable (secretly creates GPU tensors)
- CPU barrier is more reliable for synchronization

### 7.2 Graph Capture Context
**File**: `vllm/distributed/parallel_state.py` (lines 470-512)

For CUDA graph capture:

```python
@contextmanager
def graph_capture(self, graph_capture_context=None):
    if graph_capture_context is None:
        stream = torch.cuda.Stream()
        graph_capture_context = GraphCaptureContext(stream)
    
    # Prepare device communicators for capture
    maybe_ca_context = self.device_communicator.ca_comm.capture()
    
    # Ensure initialization complete before graph capture
    stream.wait_stream(torch.cuda.current_stream())
    
    with torch.cuda.stream(stream), maybe_ca_context:
        yield graph_capture_context
```

### 7.3 Health Monitoring
**File**: `vllm/v1/executor/multiproc_executor.py` (lines 267-298)

```python
def start_worker_monitor(self, inline=False):
    def monitor_workers():
        sentinels = [h.proc.sentinel for h in self.workers]
        died = multiprocessing.connection.wait(sentinels)
        
        # Worker died unexpectedly
        logger.error("Worker proc %s died, shutting down executor.", proc_name)
        self.is_failed = True
        self.shutdown()
        if self.failure_callback:
            self.failure_callback()
    
    Thread(target=monitor_workers, daemon=True).start()
```

## 8. Communication Patterns Summary

### 8.1 Input Flow (SchedulerOutput → Workers)
```
Engine
  ↓
[Rank 0] ← CollectiveRPC with SchedulerOutput
  ↓
[MessageQueue (SharedMemory)] ← Broadcast to all ranks
  ↓
[All Workers] ← Receive from SharedMemory
  ↓
Model Execution with TP Collectives
```

### 8.2 Output Flow (Workers → Engine)
```
[All Workers] ← Execute Model (TP collectives internally)
  ↓
[Response MessageQueues] ← All workers send outputs
  ↓
[Rank 0] ← Aggregates (if needed) and returns
  ↓
Engine ← Result via Future
```

### 8.3 TP Collective (Within Model Execution)
```
[TP Rank 0, 1, 2, 3] ← Forward pass at same layer
  ↓
AllReduce (via device communicator)
  ↓
Backend Selection:
  - NCCL_SYMM_MEM (if enabled, world_size fits)
  - QUICK_REDUCE (if AMD MI300)
  - FLASHINFER (if supported)
  - CUSTOM_ALLREDUCE (if NVLink available)
  - SYMM_MEM (if PyTorch 2.4+)
  - PYNCCL (fallback)
  ↓
[TP Rank 0, 1, 2, 3] ← Result combined
```

## 9. Key Optimizations

### 9.1 SharedMemory Broadcasting
- Zero-copy transmission of scheduler outputs
- Lock-free ring buffer for high throughput
- Adaptive busy-waiting vs idle mode
- Significantly reduces latency vs socket-based communication

### 9.2 Custom AllReduce
- NVLink peer-to-peer for same-node TP
- Bypasses main memory and PCIe
- Supports specific world sizes (2, 4, 6, 8)
- Automatic fallback if conditions not met

### 9.3 Pipelined Communication
- Async broadcast of scheduler outputs
- Overlap with previous model execution
- Multiple message queues for buffering

### 9.4 Backend Selection
- Automatic dispatch based on input characteristics
- Tuned parameters per world size
- Graceful fallback chain

### 9.5 Single-Result Return
- Only one worker (output_rank) sends result
- Reduces network traffic
- KV cache aggregation handled separately

## 10. Configuration Options

### Environment Variables
- `VLLM_ALLREDUCE_USE_SYMM_MEM`: Enable symmetric memory all-reduce
- `VLLM_ALLREDUCE_USE_FLASHINFER`: Enable FlashInfer all-reduce
- `VLLM_SKIP_P2P_CHECK`: Trust driver's P2P report for custom all-reduce
- `VLLM_BATCH_INVARIANT`: Disable batch-invariant optimization
- `VLLM_MQ_MAX_CHUNK_BYTES_MB`: Message queue chunk size
- `VLLM_DISABLE_PYNCCL`: Disable PyNCCL communicator
- `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`: Use Ray V2 executor

### Config Objects
- `parallel_config.tensor_parallel_size`: TP degree
- `parallel_config.distributed_executor_backend`: "mp", "ray", "uni", etc.
- `parallel_config.worker_cls`: Worker class (qualified name string)
- `parallel_config.placement_group`: Ray placement group

## 11. Troubleshooting Guide

### Custom AllReduce Disabled Issues
**Possible causes**:
1. Multi-node TP (check: `in_the_same_node_as` fails)
2. Unsupported world size (only [2,4,6,8] supported)
3. No NVLink P2P access (check: `gpu_p2p_access_check` fails)
4. Missing custom_ops library (non-GPU environment)

### AllReduce Performance
**Optimization priority**:
1. If symmetric memory available & batch_invariant=False → NCCL_SYMM_MEM
2. If AMD MI300 → QUICK_REDUCE
3. If FlashInfer available & applicable → FLASHINFER
4. If NVLink & single-node → CUSTOM_ALLREDUCE
5. Otherwise → PYNCCL

### Message Queue Delays
**Check**:
1. Process affinity conflicts (OpenMP, NUMA)
2. CPU contention (check CPU utilization)
3. Memory bandwidth saturation
4. Large batch sizes with small `busy_loop_s`

## 12. Code Flow Example: TP=2 Execution

```
Engine.generate()
  ↓
executor.execute_model(scheduler_output)
  ↓
[Rank 0] collective_rpc("execute_model", scheduler_output)
  ↓
rpc_broadcast_mq.send(scheduler_output)  # SharedMemory write
  ↓
[All Ranks] rpc_broadcast_mq.recv()  # SharedMemory read + spinwait
  ↓
[Rank 0 & 1] worker.execute_model(scheduler_output)
  ↓
model.forward()
  ↓
[At TP layer] hidden_states = output  # Different splits on each rank
  ↓
tp_group.all_reduce(hidden_states)
  ↓
[Device Communicator] all_reduce dispatches to appropriate backend
  ↓
[Rank 0 & 1] hidden_states = combined result
  ↓
Continue forward pass...
  ↓
[Rank 0] return output to executor
[Rank 1] (no return)
  ↓
executor.result()
```

