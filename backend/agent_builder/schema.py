#
# Agent schema — the declarative contract the Phase 2 Copilot reads and writes.
#
# Design rule: stay as close to Pipecat Flows' own vocabulary as possible. A node
# carries Pipecat's native fields (`role_message`, `task_messages`, `pre/post_actions`)
# verbatim. The ONLY thing we add is `edges`: transitions expressed as DATA (a string
# `target`) rather than as Python closures — because a Copilot can emit a string, not
# a callable. `AgentBuilder` turns these strings back into the closures Pipecat wants.
#
# Context-management additions (also plain data, so the builder UI can edit them):
#   Node.tools    — names of registered lookup tools (e.g. find_appointment_types)
#                   the LLM may call while on this node. Tools never transition.
#   Node.context  — "append" (default) or "reset": on entering the node, drop the
#                   transcript and carry over only a compact summary of state.
#   Node.prefetch — tools the SERVER runs on entering the node (args may reference
#                   state as "$key"); results are injected into the node's context,
#                   saving a full LLM round trip per lookup (latency on voice).
#   Edge.action   — name of a registered server-side guard run before the
#                   transition (e.g. select_service). If it fails, the LLM stays on
#                   the node and receives the reason + alternatives to self-correct.
#

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs "Rachel"
DEFAULT_MODEL = "gpt-4o"


@dataclass
class Edge:
    """A transition out of a node, exposed to the LLM as a callable tool."""

    function: str            # tool name the LLM calls to take this edge
    description: str         # when the model should call it
    target: str              # node to transition to (by name)
    # Fields to collect on this edge, as JSON-schema properties.
    properties: dict = field(default_factory=dict)
    required: list = field(default_factory=list)
    action: Optional[str] = None            # server-side guard/effect run before transitioning
    fallback_target: Optional[str] = None   # where to go if the action fails (default: stay)

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            function=d["function"],
            description=d["description"],
            target=d["target"],
            properties=d.get("properties", {}),
            required=d.get("required", []),
            action=d.get("action"),
            fallback_target=d.get("fallback_target"),
        )


@dataclass
class Node:
    """A single conversational state. Fields mirror Pipecat Flows' NodeConfig."""

    name: str
    task_messages: list = field(default_factory=list)   # this node's objectives
    role_message: Optional[str] = None                  # overrides the global persona
    edges: list = field(default_factory=list)           # list[Edge]; transitions out
    pre_actions: list = field(default_factory=list)
    post_actions: list = field(default_factory=list)
    end: bool = False                                   # terminal -> ends the call
    tools: list = field(default_factory=list)           # lookup tools available on this node
    context: str = "append"                             # "append" | "reset" (summary carry-over)
    prefetch: list = field(default_factory=list)        # [{"tool": name, "args": {k: "$state_key"|value}}]

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            name=d["name"],
            task_messages=d.get("task_messages", []),
            role_message=d.get("role_message"),
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
            pre_actions=d.get("pre_actions", []),
            post_actions=d.get("post_actions", []),
            end=d.get("end", False),
            tools=d.get("tools", []),
            context=d.get("context", "append"),
            prefetch=d.get("prefetch", []),
        )


@dataclass
class AgentConfig:
    """A complete agent: identity + the conversation graph."""

    name: str
    initial_node: str
    nodes: list                          # list[Node]
    persona: str = ""                    # global role_message, applied to every node
    voice_id: str = DEFAULT_VOICE_ID
    model: str = DEFAULT_MODEL

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        return cls(
            name=d["name"],
            initial_node=d["initial_node"],
            nodes=[Node.from_dict(n) for n in d["nodes"]],
            persona=d.get("persona", ""),
            voice_id=d.get("voice_id", DEFAULT_VOICE_ID),
            model=d.get("model", DEFAULT_MODEL),
        )
