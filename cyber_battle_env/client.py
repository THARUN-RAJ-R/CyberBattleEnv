"""
CyberBattleEnv HTTP/WebSocket client.

Usage (sync, e.g. notebooks):
    with CyberBattleEnv(base_url="https://your-space.hf.space").sync() as env:
        obs = env.reset(task="easy")
        result = env.step(CyberBattleAction(action_type="scan", target_node=1))

Usage (async):
    async with CyberBattleEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset(task="medium")
        result = await env.step(CyberBattleAction(action_type="exploit", target_node=1))

Usage (from Docker image):
    env = await CyberBattleEnv.from_docker_image("cyber-battle-env:latest")
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from typing import Any, Dict, Optional

import httpx

from .models import (
    CyberBattleAction,
    CyberBattleObservation,
    CyberBattleState,
)

logger = logging.getLogger(__name__)

# Try openenv-core client base; fall back to our own thin wrapper
try:
    from openenv.core.env_client import EnvClient  # type: ignore
    from openenv.core.client_types import StepResult  # type: ignore

    class CyberBattleEnv(EnvClient):  # type: ignore[misc]
        """openenv-core powered client (WebSocket)."""

        def _step_payload(self, action: CyberBattleAction) -> dict:
            return {
                "action_type": action.action_type,
                "target_node": action.target_node,
                "parameters": action.parameters,
            }

        def _parse_result(self, payload: dict) -> StepResult:
            obs_data = payload.get("observation", payload)
            obs = CyberBattleObservation(**obs_data)
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict) -> CyberBattleState:
            return CyberBattleState(**payload)

    _USING_OPENENV_CORE = True
    logger.info("CyberBattleEnv client using openenv-core WebSocket transport")

except ImportError:
    _USING_OPENENV_CORE = False

    class _StepResult:  # type: ignore[no-redef]
        """Lightweight StepResult used when openenv-core is not installed."""

        def __init__(
            self,
            observation: CyberBattleObservation,
            reward: Optional[float],
            done: bool,
        ) -> None:
            self.observation = observation
            self.reward = reward
            self.done = done

    class _SyncWrapper:
        """Synchronous context-manager wrapper around the async client."""

        def __init__(self, async_client: "CyberBattleEnv") -> None:
            self._client = async_client
            self._loop = asyncio.new_event_loop()

        def __enter__(self):
            self._loop.run_until_complete(self._client.__aenter__())
            return self

        def __exit__(self, *args):
            self._loop.run_until_complete(self._client.__aexit__(*args))
            self._loop.close()

        def reset(self, **kwargs) -> _StepResult:
            return self._loop.run_until_complete(self._client.reset(**kwargs))

        def step(self, action: CyberBattleAction) -> _StepResult:
            return self._loop.run_until_complete(self._client.step(action))

        def state(self) -> CyberBattleState:
            return self._loop.run_until_complete(self._client.state())

        def close(self):
            self._loop.run_until_complete(self._client.close())

    class CyberBattleEnv:  # type: ignore[no-redef]
        """
        Fallback HTTP client (no openenv-core dependency).
        Implements async context manager protocol.
        """

        def __init__(self, base_url: str = "http://localhost:8000") -> None:
            self.base_url = base_url.rstrip("/")
            self._http: Optional[httpx.AsyncClient] = None
            self._container_id: Optional[str] = None

        # ── Lifecycle ──────────────────────────────────────────────────────

        async def __aenter__(self) -> "CyberBattleEnv":
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
            return self

        async def __aexit__(self, *_) -> None:
            await self.close()

        async def close(self) -> None:
            if self._http:
                await self._http.aclose()
                self._http = None
            if self._container_id:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    check=False, capture_output=True,
                )
                logger.info("Stopped container %s", self._container_id)

        def sync(self) -> _SyncWrapper:
            """Return a synchronous wrapper (for notebooks / scripts)."""
            return _SyncWrapper(self)

        # ── OpenEnv interface ──────────────────────────────────────────────

        async def reset(self, task: str = "easy", seed: Optional[int] = None, **kwargs) -> _StepResult:
            assert self._http is not None, "Use inside 'async with' block"
            payload = {"task": task}
            if seed is not None:
                payload["seed"] = seed
            r = await self._http.post("/reset", json=payload)
            r.raise_for_status()
            obs = CyberBattleObservation(**r.json())
            return _StepResult(obs, obs.reward, obs.done)

        async def step(self, action: CyberBattleAction) -> _StepResult:
            assert self._http is not None, "Use inside 'async with' block"
            payload = {
                "action_type": action.action_type,
                "target_node": action.target_node,
                "parameters": action.parameters,
            }
            r = await self._http.post("/step", json=payload)
            r.raise_for_status()
            obs = CyberBattleObservation(**r.json())
            return _StepResult(obs, obs.reward, obs.done)

        async def state(self) -> CyberBattleState:
            assert self._http is not None, "Use inside 'async with' block"
            r = await self._http.get("/state")
            r.raise_for_status()
            return CyberBattleState(**r.json())

        # ── Docker factory ─────────────────────────────────────────────────

        @classmethod
        async def from_docker_image(
            cls,
            image_name: str,
            port: int = 8000,
            startup_wait: float = 5.0,
            extra_env: Optional[Dict[str, str]] = None,
        ) -> "CyberBattleEnv":
            """
            Spin up a Docker container and return a connected client.

            Args:
                image_name:   Docker image tag, e.g. "cyber-battle-env:latest"
                port:         Host port to map container's 8000.
                startup_wait: Seconds to wait for server to become healthy.
                extra_env:    Extra env vars to pass to the container.
            """
            cmd = [
                "docker", "run", "-d",
                "-p", f"{port}:8000",
            ]
            if extra_env:
                for k, v in extra_env.items():
                    cmd += ["-e", f"{k}={v}"]
            cmd.append(image_name)

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            logger.info("Started container %s from image %s", container_id[:12], image_name)

            base_url = f"http://localhost:{port}"
            client = cls(base_url=base_url)
            client._container_id = container_id
            client._http = httpx.AsyncClient(base_url=base_url, timeout=30.0)

            # Wait for health endpoint
            deadline = time.time() + startup_wait + 30
            while time.time() < deadline:
                try:
                    r = await client._http.get("/health")
                    if r.status_code == 200:
                        logger.info("Container healthy at %s", base_url)
                        return client
                except Exception:
                    pass
                await asyncio.sleep(1.0)

            raise RuntimeError(f"Container {container_id[:12]} did not become healthy in time")

    logger.info("CyberBattleEnv client using HTTP fallback transport")
