"""
FusionOps V2 Client
OpenEnv-compatible client for the FusionOps V2 scheduling environment.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from pydantic import BaseModel


class FusionOpsObservation(BaseModel):
    text: str
    error: Optional[str] = None


class FusionOpsAction(BaseModel):
    command: str  # JSON action string


class FusionOpsResult(BaseModel):
    observation: FusionOpsObservation
    reward: float = 0.0
    done: bool = False
    score: Optional[float] = None
    info: dict = {}


class FusionOpsEnv:
    """Client for the FusionOps V2 environment server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None
        self._last_score: Optional[float] = None

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None):
        """
        Start a Docker container from the given image and wait for it to be ready.
        If image_name is None, assumes server is already running on localhost:7860.
        """
        if not image_name:
            return cls(base_url="http://localhost:7860")

        import subprocess, time, aiohttp
        port = 7860
        container_name = f"fusionops-{int(time.time())}"

        # Start container
        subprocess.run(
            ["docker", "run", "-d", "--rm",
             "--name", container_name,
             "-p", f"{port}:{port}",
             image_name],
            check=True, capture_output=True,
        )

        # Wait for server to be ready (up to 30s)
        base_url = f"http://localhost:{port}"
        for attempt in range(30):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{base_url}/",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status == 200:
                            break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            # Cleanup on failure
            subprocess.run(["docker", "stop", container_name],
                           capture_output=True)
            raise RuntimeError(
                f"Container {container_name} did not become ready in 30s"
            )

        instance = cls(base_url=base_url)
        instance._container_name = container_name
        return instance

    async def reset(self, task: str = "task1_chain") -> FusionOpsResult:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/reset",
                json={"task": task},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                self._session_id = data.get("session_id")
                return FusionOpsResult(
                    observation=FusionOpsObservation(
                        text=data.get("observation", ""),
                    ),
                    reward=data.get("reward", 0.0),
                    done=data.get("done", False),
                )

    async def step(self, action: FusionOpsAction) -> FusionOpsResult:
        import aiohttp
        if self._session_id is None:
            raise RuntimeError("Must call reset() first")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/step/{self._session_id}",
                json={"command": action.command},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                info = data.get("info", {})
                score = data.get("score")
                if score is not None:
                    self._last_score = score
                return FusionOpsResult(
                    observation=FusionOpsObservation(
                        text=data.get("observation", ""),
                        error=info.get("error"),
                    ),
                    reward=data.get("reward", 0.0),
                    done=data.get("done", False),
                    score=score,
                    info=info,
                )

    async def state(self) -> dict:
        import aiohttp
        if self._session_id is None:
            raise RuntimeError("Must call reset() first")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/state/{self._session_id}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return await resp.json()

    async def close(self):
        self._session_id = None
        # Stop Docker container if we started one
        container = getattr(self, "_container_name", None)
        if container:
            import subprocess
            subprocess.run(
                ["docker", "stop", container],
                capture_output=True,
            )
            self._container_name = None

    def get_score(self) -> float:
        return self._last_score or 0.0
