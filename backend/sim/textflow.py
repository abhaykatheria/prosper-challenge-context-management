#
# TextFlowRunner — runs a compiled agent graph over text, without audio.
#
# It mirrors the Pipecat Flows semantics the voice bot relies on (role message as
# system instruction, task messages appended per node, APPEND/RESET context
# strategies, function handlers returning (result, next_node)) so the exact same
# AgentBuilder output can be exercised cheaply: by the eval harness, and by the
# builder UI's text-chat test mode. Every LLM call's token usage is recorded.
#

import json
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI
from pipecat_flows import ContextStrategy

# USD per 1M tokens (input, cached input, output). OpenAI list prices.
PRICES = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    max_prompt_tokens: int = 0
    per_call: list = field(default_factory=list)

    def add(self, u, node: str):
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        self.calls += 1
        self.prompt_tokens += u.prompt_tokens
        self.cached_tokens += cached
        self.completion_tokens += u.completion_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, u.prompt_tokens)
        self.per_call.append({"node": node, "prompt": u.prompt_tokens, "cached": cached,
                              "completion": u.completion_tokens})

    def cost(self, model: str, use_cache: bool = True) -> float:
        inp, cached, out = PRICES.get(model, PRICES["gpt-4.1"])
        if not use_cache:
            return (self.prompt_tokens * inp + self.completion_tokens * out) / 1e6
        return ((self.prompt_tokens - self.cached_tokens) * inp + self.cached_tokens * cached
                + self.completion_tokens * out) / 1e6


class TextFlowRunner:
    def __init__(self, builder, client: Optional[AsyncOpenAI] = None, model: Optional[str] = None):
        self.builder = builder
        self.client = client or AsyncOpenAI()
        self.model = model or builder.config.model
        self.state: dict = {}
        self.messages: list = []
        self.system = ""
        self.functions: dict = {}
        self.node_name = ""
        self.ended = False
        self._end_after_reply = False
        self.usage = Usage()
        self.transcript: list = []  # [{"role": "assistant"|"user"|"event", ...}]

    # ---- node handling (mirrors FlowManager) --------------------------------
    def _set_node(self, node: dict):
        self.node_name = node.get("name", "")
        if node.get("role_message"):
            self.system = node["role_message"]
        strategy = node.get("context_strategy")
        if strategy is not None and strategy.strategy == ContextStrategy.RESET:
            self.messages = []
        self.messages.extend(node.get("task_messages", []))
        self.functions = {f.name: f for f in node.get("functions", [])}
        self._end_after_reply = any(a.get("type") == "end_conversation" for a in node.get("post_actions", []))
        self.transcript.append({"role": "event", "text": f"-> node {self.node_name}"})

    def _tool_schemas(self) -> list:
        return [
            {"type": "function", "function": {
                "name": f.name, "description": f.description,
                "parameters": {"type": "object", "properties": f.properties, "required": f.required}}}
            for f in self.functions.values()
        ]

    # ---- conversation -------------------------------------------------------
    async def start(self) -> str:
        self._set_node(self.builder.build_initial_node())
        return await self._run_llm()

    async def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self.transcript.append({"role": "user", "text": user_text})
        return await self._run_llm()

    async def _run_llm(self, max_steps: int = 8) -> str:
        for _ in range(max_steps):
            kwargs = {}
            tools = self._tool_schemas()
            if tools:
                kwargs["tools"] = tools
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system}] + self.messages,
                temperature=0.2,
                **kwargs,
            )
            self.usage.add(resp.usage, self.node_name)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                text = msg.content or ""
                self.messages.append({"role": "assistant", "content": text})
                self.transcript.append({"role": "assistant", "text": text})
                if self._end_after_reply:
                    self.ended = True
                return text

            self.messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            next_node = None
            for tc in msg.tool_calls:
                fn = self.functions.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if fn is None:
                    result = {"status": "error", "message": f"Unknown function {tc.function.name}"}
                else:
                    result, nn = await fn.handler(args, self)
                    next_node = nn or next_node
                self.transcript.append({"role": "event", "text": f"{tc.function.name}({json.dumps(args)}) -> "
                                        f"{json.dumps(result)[:300]}"})
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            if next_node is not None:
                self._set_node(next_node)
        return ""
