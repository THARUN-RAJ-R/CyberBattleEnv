"""
FastAPI server for CyberBattleEnv.

Endpoints:
  POST /reset          — Start new episode
  POST /step           — Take attacker action
  GET  /state          — Current episode metadata
  GET  /health         — Health check
  GET  /docs           — Auto-generated OpenAPI docs

Tries to use create_fastapi_app() from openenv-core for WebSocket /ws support.
Falls back to a raw FastAPI implementation so the server always works.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .environment import CyberBattleEnvironment
from ..models import CyberBattleAction, CyberBattleObservation, CyberBattleState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Attempt openenv-core integration ─────────────────────────────────────────
_openenv_app = None
try:
    from openenv.core.env_server import create_fastapi_app  # type: ignore
    _openenv_app = create_fastapi_app(CyberBattleEnvironment)
    logger.info("Using openenv-core create_fastapi_app (WebSocket /ws enabled)")
except Exception as _e:
    logger.info("openenv-core not available (%s) — using raw FastAPI implementation", _e)


# ── Request/response wrappers ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    role: str = "attacker"   # "attacker" | "defender"


class StepRequest(BaseModel):
    action_type: str
    target_node: int = 1
    parameters: Dict[str, Any] = {}
    role: Optional[str] = None       # attacker|defender recovery hint
    last_task: Optional[str] = None  # task recovery hint


# ── Singleton environment (one per container process) ─────────────────────────
# For concurrent sessions openenv-core manages per-WebSocket instances.
# The HTTP fallback uses a single in-memory env (suitable for single-agent runs).
_env = CyberBattleEnvironment()


# ── Raw FastAPI app ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("CyberBattleEnv server starting")
    yield
    logger.info("CyberBattleEnv server shutting down")


def _make_app() -> FastAPI:
    application = FastAPI(
        title="CyberBattleEnv",
        description=(
            "Multi-agent cybersecurity RL environment. "
            "Simulates enterprise network: User → Web Server → App Server → Database. "
            "Attacker vs scripted Defender across 3 difficulty tasks."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──────────────────────────────────────────────────────────────

    @application.get("/health", tags=["meta"])
    async def health():
        return {"status": "healthy", "env": "cyber-battle-env", "version": "1.0.0"}

    # ── Reset ───────────────────────────────────────────────────────────────

    @application.post("/reset", response_model=CyberBattleObservation, tags=["env"])
    async def reset(req: ResetRequest = ResetRequest()):
        try:
            obs = _env.reset(task=req.task, seed=req.seed, episode_id=req.episode_id, role=req.role)
            return obs
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # ── Step ────────────────────────────────────────────────────────────────

    @application.post("/step", response_model=CyberBattleObservation, tags=["env"])
    async def step(req: StepRequest):
        # Auto-recover from multi-worker state loss on HF Spaces.
        # If env was reset by a different worker, the role may have defaulted back
        # to "attacker". The client sends role+task so we can re-apply it.
        if req.role and _env._role != req.role:
            _env._role = req.role
        if req.last_task and _env._task != req.last_task:
            # Full re-reset needed (different worker got the original reset)
            _env.reset(task=req.last_task, role=req.role or "attacker", seed=42)

        action = CyberBattleAction(
            action_type=req.action_type,
            target_node=req.target_node,
            parameters=req.parameters,
        )
        obs = _env.step(action)
        return obs


    # ── State ────────────────────────────────────────────────────────────────

    @application.get("/state", response_model=CyberBattleState, tags=["env"])
    async def state():
        return _env.state

    # ── Defender Step (AI vs AI — hard task) ─────────────────────────────────
    # Allows the LLM defender to take an action in the environment.
    # In the hard task, inference.py calls this after each attacker step,
    # making it TRUE AI vs AI: both sides controlled by the LLM.

    @application.post("/defender_step", response_model=CyberBattleObservation, tags=["env"])
    async def defender_step(req: StepRequest):
        from ..models import DefenderActionType
        valid = [a.value for a in DefenderActionType]
        if req.action_type not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid defender action '{req.action_type}'. Valid: {valid}",
            )
        obs = _env.apply_defender_action(req.action_type, req.target_node)
        return obs


    # ── Web UI stub ──────────────────────────────────────────────────────────

    @application.get("/web", tags=["meta"])
    async def web():
        html = """
        <html><head><title>CyberBattleEnv</title></head><body>
        <h1>CyberBattleEnv</h1>
        <p>Multi-agent cybersecurity RL environment.</p>
        <ul>
          <li><a href="/docs">API Docs (Swagger)</a></li>
          <li><a href="/health">Health Check</a></li>
          <li><a href="/state">Current State</a></li>
        </ul>
        <h2>Quick Start</h2>
        <pre>curl -X POST /reset -H "Content-Type: application/json" -d '{"task":"easy"}'</pre>
        </body></html>
        """
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html)

    # ── Info ─────────────────────────────────────────────────────────────────

    @application.get("/", tags=["meta"])
    async def root():
        return {
            "name": "CyberBattleEnv",
            "version": "1.0.0",
            "tasks": ["easy", "medium", "hard"],
            "endpoints": ["/reset", "/step", "/state", "/health", "/web", "/docs"],
        }

    return application


# ── Export `app` ──────────────────────────────────────────────────────────────
# Use openenv-core app if available (WebSocket support), otherwise raw FastAPI.
app: FastAPI = _openenv_app if _openenv_app is not None else _make_app()

# Ensure the raw-FastAPI routes always exist (openenv-core may not expose /web etc.)
if _openenv_app is not None:
    _raw = _make_app()
    # Merge non-conflicting routes from raw app into openenv app
    existing_paths = {r.path for r in _openenv_app.routes}
    for route in _raw.routes:
        if hasattr(route, "path") and route.path not in existing_paths:
            _openenv_app.routes.append(route)


def main() -> None:
    """Entry point for `openenv validate` and `[project.scripts]`."""
    import uvicorn
    uvicorn.run(
        "cyber_battle_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
