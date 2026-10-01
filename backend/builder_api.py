#
# Builder API — the minimal agent-builder UI's backend, mounted on the Pipecat
# dev runner's FastAPI app (so the builder, the prebuilt call client and the bot
# share one server and one port).
#
#   /builder                    static single-page editor (ui/builder.html)
#   /builder/api/agents         list / create agents (JSON files in agents/)
#   /builder/api/agents/{id}    read / save (validated by AgentBuilder) / delete
#   /builder/api/agents/{id}/activate   the agent the next test call will run
#   /builder/api/meta           registered tools + actions (for the editor)
#   /builder/api/trace          live call trace: nodes, tools, guards, token usage
#   /builder/api/catalog/search retrieval playground over the CatalogEngine
#

import json
import re
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from agent_builder import AgentBuilder, AgentConfig
from catalog.tools import ACTIONS, TOOLS, get_engine

HERE = Path(__file__).parent
AGENTS_DIR = HERE / "agents"
UI_FILE = HERE / "ui" / "builder.html"
ACTIVE_FILE = AGENTS_DIR / ".active"
DEFAULT_AGENT = "scheduler"

# ---- live trace (in-memory ring buffer; one dev server, one demo) -----------
_trace: deque = deque(maxlen=500)
_seq = 0


LOG_FILE = HERE / "logs" / "trace.jsonl"


def emit(event: dict) -> None:
    global _seq
    _seq += 1
    e = {"seq": _seq, "ts": time.time(), **event}
    _trace.append(e)
    try:  # persisted so a call can be analysed after a server restart
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(e, default=str) + "\n")
    except OSError:
        pass


def active_agent_path() -> Path:
    name = ACTIVE_FILE.read_text().strip() if ACTIVE_FILE.exists() else DEFAULT_AGENT
    path = AGENTS_DIR / f"{name}.json"
    return path if path.exists() else AGENTS_DIR / f"{DEFAULT_AGENT}.json"


def _path(agent_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", agent_id):
        raise HTTPException(400, "Invalid agent id")
    return AGENTS_DIR / f"{agent_id}.json"


def _validate(data: dict) -> list:
    try:
        return AgentBuilder(AgentConfig.from_dict(data), validate=False).validation_errors()
    except (KeyError, TypeError) as e:
        return [f"Malformed agent: missing or invalid field {e}"]


def register(app: FastAPI) -> None:
    @app.get("/builder", include_in_schema=False)
    async def builder_page():
        return FileResponse(UI_FILE)

    @app.get("/builder/api/agents")
    async def list_agents():
        active = active_agent_path().stem
        out = []
        for p in sorted(AGENTS_DIR.glob("*.json")):
            d = json.loads(p.read_text())
            out.append({"id": p.stem, "name": d.get("name", p.stem), "nodes": len(d.get("nodes", [])),
                        "active": p.stem == active})
        return out

    @app.get("/builder/api/agents/{agent_id}")
    async def get_agent(agent_id: str):
        p = _path(agent_id)
        if not p.exists():
            raise HTTPException(404, "Not found")
        return json.loads(p.read_text())

    @app.put("/builder/api/agents/{agent_id}")
    async def save_agent(agent_id: str, data: dict):
        errors = _validate(data)
        if errors:
            raise HTTPException(422, {"errors": errors})
        _path(agent_id).write_text(json.dumps(data, indent=2))
        return {"ok": True}

    @app.post("/builder/api/agents")
    async def create_agent(body: dict):
        agent_id = re.sub(r"[^a-z0-9_-]+", "_", body.get("id", "").lower()).strip("_") or f"agent_{int(time.time())}"
        p = _path(agent_id)
        if p.exists():
            raise HTTPException(409, "Agent already exists")
        template = body.get("from")
        if template:
            data = json.loads(_path(template).read_text())
            data["name"] = body.get("name") or f"{data['name']} (copy)"
        else:
            data = {
                "name": body.get("name") or agent_id,
                "persona": "You are a friendly assistant. Your replies are spoken aloud; keep them short.",
                "initial_node": "start",
                "nodes": [{"name": "start", "task_messages": [
                    {"role": "developer", "content": "Greet the caller and ask how you can help."}]}],
            }
        p.write_text(json.dumps(data, indent=2))
        return {"id": agent_id}

    @app.delete("/builder/api/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        if agent_id == DEFAULT_AGENT:
            raise HTTPException(400, "The default agent can't be deleted")
        _path(agent_id).unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/builder/api/agents/{agent_id}/activate")
    async def activate(agent_id: str):
        if not _path(agent_id).exists():
            raise HTTPException(404, "Not found")
        ACTIVE_FILE.write_text(agent_id)
        emit({"type": "info", "text": f"Active agent set to '{agent_id}'"})
        return {"ok": True}

    @app.get("/builder/api/meta")
    async def meta():
        return {
            "tools": {k: {"description": t.description, "properties": t.properties} for k, t in TOOLS.items()},
            "actions": {k: (f.__doc__ or "").strip() for k, f in ACTIONS.items()},
        }

    @app.get("/builder/api/trace")
    async def trace(since: int = 0):
        return [e for e in _trace if e["seq"] > since]

    @app.get("/builder/api/catalog/search")
    async def catalog_search(q: str, new_patient: bool | None = None):
        e = get_engine()
        return {
            "appointment_types": e.match_appointment_types(q, new_patient=new_patient),
            "providers": e.find_providers(name=q),
            "locations": e.find_locations(q),
        }
