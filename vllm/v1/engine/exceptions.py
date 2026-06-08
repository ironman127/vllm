# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 【模块说明】exceptions —— 引擎异常类型定义
#
# 定义 v1 engine 的两类异常，用于区分可恢复与不可恢复的错误：
#
#   EngineGenerateError : 可恢复错误，由 AsyncLLM.generate() 抛出，
#                         通常是单次请求级别的错误（如 KV 加载失败），
#                         调用方可选择重试。
#   EngineDeadError     : 不可恢复错误，在 EngineCore 进程崩溃时抛出，
#                         引擎无法继续服务任何新请求；suppress_context 参数
#                         可用于屏蔽底层 ZMQError 以保持堆栈清晰。

class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
