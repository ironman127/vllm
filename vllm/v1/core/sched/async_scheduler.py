# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【模块说明】sched/async_scheduler.py —— 异步调度器（AsyncScheduler）
#
# Scheduler 的薄封装子类，专用于异步执行模式（EngineCore 带 batch_queue 时）。
# 异步模式下，调度与 GPU 执行重叠：当前步 schedule() 与上一步的 GPU forward pass
# 并发进行，GPU 结果通过 update_from_output() 异步回填。
#
# 核心差异（相对 Scheduler 基类）：
#
#   num_output_placeholders    : 每个请求在飞中（in-flight）的待填充 token 槽位数，
#                                用于跟踪"已调度但尚未收到 GPU 结果"的状态。
#
#   _update_after_schedule()   : 覆盖基类，在调度后对所有非 prefill_chunk 请求
#                                将 num_output_placeholders 加 1，并设置
#                                pending_structured_output_tokens 标志。
#
#   _update_request_with_output(): 覆盖基类，处理异步结果回填时的保护逻辑：
#                                - async_tokens_to_discard > 0 时，该请求已被抢占，
#                                  静默丢弃该 GPU 输出并递减计数器，不推进请求状态；
#                                - 否则正常调用基类逻辑，并递减 num_output_placeholders。
#
# 抢占保护机制：
#   若请求在 in-flight 期间被抢占，基类 _preempt_request() 会设置
#   request.async_tokens_to_discard = num_output_placeholders，
#   AsyncScheduler 在收到 GPU 结果时识别该标记并丢弃结果，保证状态一致性。

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.async_tokens_to_discard > 0:
            # The request was force-preempted in reset_prefix_cache; drop one
            # stale in-flight async output frame per call until the counter
            # is drained.
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
