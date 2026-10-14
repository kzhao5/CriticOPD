# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from .single_turn_agent_loop import SingleTurnAgentLoop
from .tool_agent_loop import ToolAgentLoop
# CriticOPD:@register 只在模块被导入时才执行。AgentLoopWorker 不做自动扫描,只依赖这里的
# 显式导入,所以不加这一行就会在 generate_sequences 里抛
# "Agent loop critic_opd_agent not registered"(job 13937168 就是这么死的)。
from .critic_opd_agent_loop import CriticOpdAgentLoop

_ = [SingleTurnAgentLoop, ToolAgentLoop, CriticOpdAgentLoop]

__all__ = [
    "AgentLoopBase",
    "AgentLoopManager",
    "AgentLoopWorker",
    "AgentLoopOutput",
    "get_trajectory_info",
]
