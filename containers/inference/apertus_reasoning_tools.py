# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""vLLM reasoning parser ``apertus_tools``: Apertus thinking that a tool call ends.

With thinking on, Apertus-v1.5 opens ``<|inner_prefix|>`` and emits its tool call as
``<|tools_prefix|>[...]`` WITHOUT closing the block first. vLLM's ``apertus`` parser then labels the
whole output reasoning, the tool parser never sees the call, and an agent ends after one turn. This
parser treats ``<|tools_prefix|>`` as an end of reasoning that stays in the content, where the
``apertus`` tool parser reads it. Loaded with ``--reasoning-parser-plugin <this file>
--reasoning-parser apertus_tools`` ([hardware.gh200.models.apertus15]).
"""

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.apertus_reasoning_parser import ApertusReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

TOOLS_PREFIX = "<|tools_prefix|>"


@ReasoningParserManager.register_module("apertus_tools")
class ApertusToolsReasoningParser(ApertusReasoningParser):
    """:class:`ApertusReasoningParser` whose reasoning also ends at :data:`TOOLS_PREFIX`."""

    @cached_property
    def tools_token_id(self) -> int | None:
        return self.vocab.get(TOOLS_PREFIX)

    def ends(self, token_id: int) -> bool:
        return token_id == self.end_token_id or (self.tools_token_id is not None and token_id == self.tools_token_id)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if self.ends(token_id):
                return True
        return False

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Sequence[int]) -> bool:
        return any(self.ends(token_id) for token_id in delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        for index, token_id in enumerate(input_ids[:-1]):
            if token_id == self.end_token_id:
                return input_ids[index + 1 :]
            if token_id == self.tools_token_id:
                return input_ids[index:]
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        if self.end_token not in model_output and TOOLS_PREFIX in model_output:
            before, _, after = model_output.partition(TOOLS_PREFIX)
            reasoning = before.partition(self.start_token)[2] if self.start_token in before else before
            return reasoning or None, TOOLS_PREFIX + after
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        tools = self.tools_token_id
        if tools is not None and self.end_token_id not in previous_token_ids:
            if tools in previous_token_ids:
                return DeltaMessage(content=delta_text)
            if tools in delta_token_ids and TOOLS_PREFIX in delta_text:
                before, _, after = delta_text.partition(TOOLS_PREFIX)
                reasoning = before.replace(self.start_token, "")
                return DeltaMessage(reasoning=reasoning or None, content=TOOLS_PREFIX + after)
        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids
        )
