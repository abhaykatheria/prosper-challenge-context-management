#
# AgentBuilder — loads a declarative agent (JSON / dict) and compiles its node
# graph into Pipecat Flows objects.
#
#   JSON  ->  AgentConfig (validated)  ->  Pipecat Flows NodeConfig graph
#
# This is the seam between "agent as data" (what the builder UI edits) and
# "agent as a running conversation" (what bot.py executes). Keeping the compile +
# validation here means bot.py never touches the graph internals.
#
# Beyond plain transitions, the builder wires two context-management mechanisms
# (see schema.py): node-scoped lookup tools and edge guard actions, and per-node
# context resets that carry over a compact state summary instead of the transcript.
#

import inspect
import json
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Union

from loguru import logger
from pipecat_flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)

from catalog.tools import ACTIONS, TOOLS, get_engine

from .schema import AgentConfig, Edge, Node

# State keys worth carrying across a context reset, with how to label them.
# Everything else (raw tool output, chit-chat) is deliberately dropped.
SUMMARY_KEYS = [
    ("full_name", "Caller name"),
    ("date_of_birth", "Date of birth"),
    ("new_patient", "New patient"),
    ("request", "What they asked for"),
    ("preferences", "Stated preferences"),
    ("appointment_type", "Appointment type"),
    ("appointment_type_id", "appointment_type_id"),
    ("has_referral", "Has referral"),
    ("provider", "Requested provider"),
    ("provider_id", "provider_id"),
    ("location", "Requested location"),
    ("location_id", "location_id"),
    ("slot", "Chosen slot"),
]


def summarize_state(state: dict) -> str:
    lines = []
    for key, label in SUMMARY_KEYS:
        v = state.get(key)
        if v is None or v == "":
            continue
        if isinstance(v, dict):
            v = ", ".join(f"{k}: {x}" for k, x in v.items() if k != "slot_id")
        lines.append(f"- {label}: {v}")
    return "\n".join(lines)


class AgentBuilder:
    """Builds a runnable Pipecat Flows graph from a declarative AgentConfig."""

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[dict] = None,
        actions: Optional[dict] = None,
        engine_factory: Callable = get_engine,
        on_event: Optional[Callable[[dict], None]] = None,
        today: Optional[date] = None,
        validate: bool = True,
    ):
        self.config = config
        self.tools = TOOLS if tools is None else tools
        self.actions = ACTIONS if actions is None else actions
        self.engine_factory = engine_factory
        self.on_event = on_event or (lambda e: None)
        self.today = today
        self._nodes_by_name = {n.name: n for n in config.nodes}
        if validate:
            self._validate()

    # ---- loading -----------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "AgentBuilder":
        return cls(AgentConfig.from_dict(data), **kwargs)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs) -> "AgentBuilder":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data, **kwargs)

    # ---- validation --------------------------------------------------------
    def validation_errors(self) -> list:
        errors = []
        names = set(self._nodes_by_name)
        if not names:
            return ["Agent has no nodes."]
        if self.config.resolver not in ("fuzzy", "jev", "cascade"):
            errors.append(f"Unknown resolver '{self.config.resolver}' (use 'fuzzy', 'jev' or 'cascade').")
        if self.config.initial_node not in names:
            errors.append(f"initial_node '{self.config.initial_node}' is not a defined node.")
        for node in self.config.nodes:
            if node.context not in ("append", "reset"):
                errors.append(f"Node '{node.name}' has unknown context '{node.context}'.")
            for t in node.tools:
                if t not in self.tools:
                    errors.append(f"Node '{node.name}' uses unknown tool '{t}'.")
            for pf in node.prefetch:
                if pf.get("tool") not in self.tools:
                    errors.append(f"Node '{node.name}' prefetches unknown tool '{pf.get('tool')}'.")
            for edge in node.edges:
                if edge.target not in names:
                    errors.append(f"Edge '{edge.function}' in node '{node.name}' targets unknown node '{edge.target}'.")
                if edge.fallback_target and edge.fallback_target not in names:
                    errors.append(f"Edge '{edge.function}' in node '{node.name}' has unknown fallback '{edge.fallback_target}'.")
                if edge.action and edge.action not in self.actions:
                    errors.append(f"Edge '{edge.function}' in node '{node.name}' uses unknown action '{edge.action}'.")
        return errors

    def _validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise ValueError(" ".join(errors))

    # ---- compilation -------------------------------------------------------
    async def build_initial_node(self) -> NodeConfig:
        """Return the entry NodeConfig; downstream nodes are built lazily on transition."""
        return await self._make_node(self._nodes_by_name[self.config.initial_node], {})

    async def _call(self, fn, args: dict, state: dict):
        """Run a tool/action; tools may be sync (local lookups) or async (network, e.g. Jev)."""
        state.setdefault("_resolver", self.config.resolver)
        result = fn(self.engine_factory(), args, state)
        return await result if inspect.isawaitable(result) else result

    def _role_message(self, node: Node) -> str:
        today = self.today or date.today()
        base = node.role_message or self.config.persona
        return f"{base}\nToday is {today:%A, %B} {today.day}, {today.year}."

    async def _make_node(self, node: Node, state: dict) -> NodeConfig:
        task_messages = list(node.task_messages)
        node_config: NodeConfig = {"name": node.name, "role_message": self._role_message(node)}
        if node.context == "reset":
            # Deterministic summary handoff: no extra LLM call (unlike RESET_WITH_SUMMARY),
            # and only structured facts survive, so the context stays O(1) per node.
            node_config["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
            summary = summarize_state(state)
            if summary:
                task_messages = [{"role": "developer", "content": f"Known so far in this call:\n{summary}"}] + task_messages
        task_messages += await self._run_prefetch(node, state)
        node_config["task_messages"] = task_messages
        node_config["functions"] = [self._make_tool_function(t) for t in node.tools] + [
            self._make_edge_function(node, edge) for edge in node.edges
        ]
        if node.pre_actions:
            node_config["pre_actions"] = node.pre_actions
        # Explicit post_actions win; otherwise a terminal node ends the call.
        if node.post_actions:
            node_config["post_actions"] = node.post_actions
        elif node.end:
            node_config["post_actions"] = [{"type": "end_conversation"}]
        self.on_event({"type": "node", "node": node.name, "context": node.context})
        return node_config

    async def _run_prefetch(self, node: Node, state: dict) -> list:
        """Run the node's prefetch tools server-side and hand the LLM the results.

        Without this, entering e.g. offer_slots costs one LLM call just to decide to
        call search_slots and a second to speak its result; on a voice call that's
        ~1s of dead air per lookup. Tools are local and sub-millisecond.
        """
        messages = []
        for pf in node.prefetch:
            args, missing = {}, False
            for k, v in (pf.get("args") or {}).items():
                if isinstance(v, str) and v.startswith("$"):
                    v = state.get(v[1:])
                    if v in (None, ""):
                        missing = True
                        break
                args[k] = v
            if missing:
                continue
            name = pf["tool"]
            result = await self._call(self.tools[name].fn, args, state)
            self.on_event({"type": "tool", "name": name, "args": args, "result": result, "prefetch": True})
            messages.append({"role": "developer", "content":
                             f"{name}({json.dumps(args)}) was already called for you. Result:\n{json.dumps(result)}"})
        return messages

    def _make_tool_function(self, name: str) -> FlowsFunctionSchema:
        spec = self.tools[name]

        async def handler(args: dict, flow_manager: FlowManager):
            result = await self._call(spec.fn, args, flow_manager.state)
            logger.info(f"[tool {name}] {args} -> {result.get('status')}")
            self.on_event({"type": "tool", "name": name, "args": args, "result": result})
            return result, None  # stay on the current node

        return FlowsFunctionSchema(
            name=spec.name,
            description=spec.description,
            properties=spec.properties,
            required=spec.required,
            handler=handler,
        )

    def _make_edge_function(self, node: Node, edge: Edge) -> FlowsFunctionSchema:
        async def handler(args: dict, flow_manager: FlowManager):
            result = {"status": "success", **args}
            updates = {}
            if edge.action:
                ok, result, updates = await self._call(self.actions[edge.action], args, flow_manager.state)
                self.on_event({"type": "action", "name": edge.action, "ok": ok, "args": args, "result": result})
                if not ok:
                    logger.info(f"[{edge.function}] action '{edge.action}' refused: {result}")
                    if edge.fallback_target:
                        target = self._nodes_by_name[edge.fallback_target]
                        return result, await self._make_node(target, flow_manager.state)
                    return result, None  # stay; the LLM sees why and self-corrects
            # Persist what the caller gave us so later nodes can use it.
            flow_manager.state.update({k: v for k, v in args.items() if v is not None})
            flow_manager.state.update(updates)
            logger.info(f"[{edge.function}] -> {edge.target} | collected: {args}")
            self.on_event({"type": "transition", "function": edge.function, "from": node.name,
                           "to": edge.target, "args": args})
            next_node = await self._make_node(self._nodes_by_name[edge.target], flow_manager.state)
            return result, next_node

        return FlowsFunctionSchema(
            name=edge.function,
            description=edge.description,
            properties=edge.properties,
            required=edge.required,
            handler=handler,
        )
