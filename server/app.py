"""
FusionOps V2 - OpenEnv Server
FastAPI application exposing the scheduling environment via HTTP and WebSocket.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.environment import FusionOpsEnv, parse_action
from src.graph_gen import load_task, list_tasks, generate_graph, generate_training_graph, TASKS

app = FastAPI(title="FusionOps V2", description="ML Graph Scheduling RL Environment")

# Store active sessions
sessions: dict[str, FusionOpsEnv] = {}


class ResetRequest(BaseModel):
    task: str = "task1_chain"
    # For training: generate random graph
    random: bool = False
    curriculum_level: float = 0.5
    num_ops: Optional[int] = None
    difficulty: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    command: str  # JSON action string


class ResetResponse(BaseModel):
    session_id: str
    observation: str
    done: bool = False
    reward: float = 0.0


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict = {}
    score: Optional[float] = None


# ============================================================
# HTTP Endpoints
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "environment": "fusionops-v2",
        "version": "2.0.0",
        "tasks": list_tasks(),
        "action_format": {
            "fuse_with_prev": "bool",
            "tile": "int (32|64|128|256)",
            "retain": "list[int] (node IDs)",
        },
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    if request.random:
        # Generate random training graph
        if request.num_ops and request.difficulty:
            graph = generate_graph(
                num_ops=request.num_ops,
                difficulty=request.difficulty,
                seed=request.seed,
            )
        else:
            graph = generate_training_graph(request.curriculum_level)
        cfg = {"max_steps": len(graph.nodes)}
    else:
        try:
            graph, cfg = load_task(request.task)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    env = FusionOpsEnv(graph, max_steps=cfg.get("max_steps"))
    result = env.reset()

    session_id = str(uuid.uuid4())
    sessions[session_id] = env

    return ResetResponse(
        session_id=session_id,
        observation=result.observation,
        done=result.done,
        reward=result.reward,
    )


@app.post("/step/{session_id}")
async def step(session_id: str, request: StepRequest):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    env = sessions[session_id]
    action = parse_action(request.command)

    if action is None:
        return StepResponse(
            observation='{"error":"Failed to parse action. Send JSON: {\\"fuse_with_prev\\": true, \\"tile\\": 128, \\"retain\\": []}"}',
            reward=-0.1,
            done=False,
            info={"error": "Parse error"},
        )

    result = env.step(action)

    response = StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )

    if result.done:
        response.score = env.get_score()
        del sessions[session_id]

    return response


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return sessions[session_id].get_state()


@app.get("/tasks")
async def get_tasks():
    result = {}
    for name in list_tasks():
        cfg = TASKS[name]
        result[name] = {
            "description": cfg["description"],
            "num_ops": cfg["num_ops"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
        }
    return result


@app.get("/web")
async def web_ui():
    """Simple web UI for testing."""
    return HTMLResponse("""
    <html>
    <head><title>FusionOps V2</title></head>
    <body style="font-family: monospace; max-width: 800px; margin: 40px auto;">
        <h1>FusionOps V2 - ML Graph Scheduling Environment</h1>
        <p>An RL environment where LLM agents learn to schedule ML computation graphs.</p>
        <h2>API Endpoints</h2>
        <ul>
            <li><code>POST /reset</code> - Start a new episode</li>
            <li><code>POST /step/{session_id}</code> - Take an action</li>
            <li><code>GET /state/{session_id}</code> - Get current state</li>
            <li><code>GET /tasks</code> - List available tasks</li>
        </ul>
        <h2>Action Format (JSON)</h2>
        <pre>{"fuse_with_prev": true, "tile": 128, "retain": [1, 3]}</pre>
        <h2>Key Decisions Per Step</h2>
        <ul>
            <li><b>fuse_with_prev</b>: Merge this op into the current kernel group?</li>
            <li><b>tile</b>: Tile size (32, 64, 128, or 256)</li>
            <li><b>retain</b>: Which node outputs to keep in fast memory?</li>
        </ul>
    </body>
    </html>
    """)


# ============================================================
# WebSocket Endpoint (OpenEnv compatibility)
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[FusionOpsEnv] = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "reset":
                task_name = data.get("task", "task1_chain")
                try:
                    graph, cfg = load_task(task_name)
                except ValueError as e:
                    await websocket.send_json({"error": str(e)})
                    continue

                env = FusionOpsEnv(graph, max_steps=cfg.get("max_steps"))
                result = env.reset()

                await websocket.send_json({
                    "type": "reset_result",
                    "observation": result.observation,
                    "done": result.done,
                    "reward": result.reward,
                })

            elif msg_type == "step":
                if env is None:
                    await websocket.send_json({"error": "Must reset first"})
                    continue

                command = data.get("command", "")
                action = parse_action(command)

                if action is None:
                    await websocket.send_json({
                        "type": "step_result",
                        "observation": '{"error":"Parse error"}',
                        "reward": -0.1,
                        "done": False,
                        "info": {"error": "Parse error"},
                    })
                    continue

                result = env.step(action)
                response = {
                    "type": "step_result",
                    "observation": result.observation,
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                }

                if result.done:
                    response["score"] = env.get_score()

                await websocket.send_json(response)

            elif msg_type == "state":
                if env is None:
                    await websocket.send_json({"error": "Must reset first"})
                    continue
                await websocket.send_json({
                    "type": "state_result",
                    **env.get_state(),
                })

            elif msg_type == "close":
                break
            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        pass


def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
