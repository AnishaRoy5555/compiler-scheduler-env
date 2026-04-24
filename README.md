---
title: FusionOps
emoji: ♠️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# FusionOps V2

An RL environment where LLM agents learn to schedule ML computation graphs. The agent walks through operations in topological order and makes three decisions per step: fuse with the previous kernel group, choose tile size, and select which tensors to retain in fast memory.

## Why This Matters

Every ML model compiles to a DAG of operations that must execute on hardware with a small fast scratchpad and large slow main memory. The compiler must decide which ops to fuse (eliminating intermediate transfers), how to tile the computation, and which tensors to retain between kernels. Getting this wrong means 2-10x slower execution.

## What's New in V2

**Sequential hybrid action space.** One step per node in topological order. At each node the agent outputs `{fuse_with_prev, tile, retain}`. This converts scheduling into binary segmentation + small discrete decisions that LLMs can learn under RL.

**Compact JSON observations.** No prose. Fixed schema. Full graph on step 0, then only the current node + 2-node lookahead. `future_uses` field gives the agent just enough foresight for retention decisions.

**Adversarial graph topologies.** Long skip connections, multi-reuse patterns, and bottleneck chains that force non-greedy strategies. The deliberately myopic greedy baseline misses long-range retention, creating 10-25% reward gap that RL can exploit.

**Procedural generation with curriculum.** Training graphs from 8-50 ops with controllable difficulty. Mixed topologies: linear chains, residual blocks, attention patterns, diamonds, and adversarial patterns.

**Physics-based cost model.** Roofline model: `latency = max(compute, memory) + kernel_launch`. Memory transfers dominate for pointwise ops (realistic), creating strong fusion and retention signals. Tile size significantly affects compute-op efficiency.

## Action Format

```json
{"fuse_with_prev": true, "tile": 128, "retain": [3]}
```

| Field | Type | Description |
|-------|------|-------------|
| `fuse_with_prev` | bool | Merge current op into the active kernel group? |
| `tile` | int | Tile size: 32, 64, 128, or 256 |
| `retain` | list[int] | Node IDs whose outputs to keep in fast memory |

## Observation Format

Step 0 (full graph):
```json
{
  "nodes": [{"id": 0, "op": "matmul", "shape": [256,256], "inputs": []}, ...],
  "step": 0, "current_node": 0,
  "current_group": {"node_ids": [], "tile": 128},
  "fast_mem": [], "capacity": 786432, "max_fusion": 6,
  "future_uses": {"0": 2, "1": 1},
  "total_latency": 0.0, "kernel_launches": 0,
  "greedy_baseline": 2323274.0
}
```

Steps 1+: omits full graph, shows current node info + 2-node lookahead.

## Tasks

| Task | Ops | Difficulty | What It Tests |
|------|-----|------------|---------------|
| `task1_chain` | 8 | Easy | Basic fusion |
| `task2_residual` | 12 | Medium | Retention for skip connections |
| `task3_attention` | 16 | Hard | Fan-out retention + fusion boundaries |
| `task4_mixed` | 24 | Expert | Full strategy on mixed topology |
| `task5_adversarial` | 20 | Hard | Long skips, only retention-aware wins |

## API

```
POST /reset          Start episode (body: {"task": "task1_chain"})
POST /step/{id}      Take action (body: {"command": "{...json...}"})
GET  /state/{id}     Current state
GET  /tasks          List tasks
WS   /ws             WebSocket for OpenEnv
```

For training with random graphs:
```json
POST /reset {"random": true, "curriculum_level": 0.5}
```

## Verified Properties

- Retention gap: 12-18% on skip-connection graphs
- Reward spread: 0.07-0.33 (naive-greedy gap across random graphs)
- Tile variation: 218% cost range for compute ops
- Greedy baseline deliberately myopic (immediate-only retention)
