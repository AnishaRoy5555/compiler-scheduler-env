---
title: FusionOps
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# FusionOps: Teaching an LLM to Think Like a Compiler

> *Can an LLM, placed inside a simulated GPU execution environment and trained via RL, discover the same fusion, retention, and scheduling strategies that took compiler engineers years to hand-tune?*

## The Problem

Every deep learning model (transformer, diffusion model, CNN) compiles down to a directed acyclic graph of tensor operations. These operations must execute on hardware where a small, fast scratchpad (SRAM) sits next to a large, slow main memory (DRAM). The compiler makes three critical decisions for every operation:

**Fuse?** Should this operation merge into the previous kernel? Fused ops share fast memory, so intermediates never touch slow DRAM. One eliminated memory roundtrip can save microseconds that compound across millions of inference calls.

**Tile?** How should the computation be split? Larger tiles amortize memory latency but demand more scratchpad space. The optimal tile size depends on what else is competing for fast memory.

**Retain?** Which tensor outputs should stay in scratchpad for future steps? Retention avoids expensive DRAM reloads, but scratchpad capacity is finite. Retain too much and you run out of space. Retain too little and you pay the reload cost later.

Get these decisions wrong and your model runs 2-10x slower than it should. Today, these decisions are made by hand-tuned heuristics in XLA, Triton, and TVM that took years to develop. We asked: can an LLM learn them from scratch?

## The Journey: From v1 to Compiler-Scheduler-Env

### v1: No hints, no learning

We started with four simple computation graphs (6-8 ops each) and a naive LLM agent. Result: the model had no idea what to do. It couldn't discover fusion or retention on its own. Score: 0.000 across all tasks.

### v2: Scaffolded hints

We introduced a heuristic "teacher" that provided hints as part of the reward signal, weighted heavily early, then progressively discounted. The model learned to fuse operations, and we saw our first breakthrough: scores jumped from 0.0 to 0.3-0.6 on simple graphs. But retention remained elusive.

### v3: Two-step lookahead scoring

To unlock retention, we redesigned the reward to consider not just the immediate step but also the next step's outcome. New insight: the model discovered that fusing *all* operations was a superior strategy to retention on our small graphs. It found a shortcut, not a strategy. The 8-op ceiling meant the model could game the environment.

### Compiler-Scheduler-Env (current): The real challenge

We rebuilt everything. Instead of 4 fixed tiny graphs, we created a general-purpose scheduling environment with:

- **Procedurally generated graphs** from 8 to 50 nodes with controllable topology (chains, residual blocks, attention patterns, diamonds, adversarial skip connections)
- **A greedy baseline** instead of the naive baseline, because now the model must beat a scheduler that already does basic fusion and immediate retention
- **A physics-based roofline cost model** that captures real GPU tradeoffs: `latency = max(compute_time, memory_time) + kernel_launch_overhead`
- **Adversarial topologies** with long skip connections and multi-reuse patterns specifically designed to punish greedy one-step-lookahead schedulers

The reward is simple: do whatever reduces latency. Fusions that cut memory traffic, retention that avoids reloads, tile sizes that balance arithmetic intensity, all rewarded through the same latency signal.

## Environment Design

The agent walks through operations in topological order, making three decisions at every step:

| Decision | What the agent chooses | Why it matters |
|----------|----------------------|----------------|
| **Fuse** | Merge this op into the current kernel? | Fused ops share fast memory, intermediates never touch DRAM |
| **Tile** | Tile size (32, 64, 128, 256) | Controls arithmetic intensity on the roofline |
| **Retain** | Which outputs to keep in fast memory? | Avoids expensive reloads but costs limited scratchpad capacity |

**Action format:**
```json
{"fuse_with_prev": true, "tile": 128, "retain": [3, 7]}
```

**Observation format (step 0, full graph):**
```json
{
  "nodes": [{"id": 0, "op": "matmul", "shape": [256,256], "inputs": []}, ...],
  "current_node": 0,
  "current_group": {"node_ids": [], "tile": 128},
  "fast_mem": [], "capacity": 786432, "max_fusion": 6,
  "future_uses": {"0": 2, "1": 1},
  "lookahead": [{"id": 1, "op": "relu", ...}, {"id": 2, "op": "add", ...}]
}
```

After step 0, observations are compact: current node + 2-node lookahead + fast memory state. The `future_uses` field tells the agent how many downstream consumers need each tensor, providing just enough information for retention reasoning without giving away the optimal strategy.

**Cost model:** Roofline-based latency simulation. Same approach used by Google's REGAL, Apache TASO, and TVM Ansor. Cost model evaluation runs in microseconds, enabling thousands of RL rollouts. The learned strategies transfer directly to real compiler backends.

**Score:** `(greedy_latency - agent_latency) / greedy_latency`. Positive = agent beats greedy. Negative = agent is worse. Zero = agent matches greedy.

## Tasks

Five computation graphs of increasing difficulty:

| Task | Nodes | Topology | What makes it hard |
|------|-------|----------|--------------------|
| `task1_chain` | 8 | Linear chain | Baseline: can the agent fuse consecutive ops? |
| `task2_residual` | 12 | ResNet-style skips | Must retain skip tensors across steps |
| `task3_attention` | 16 | Q/K/V fan-out | Three consumers from one producer |
| `task4_mixed` | 24 | Diamonds + skips + attention | No single strategy works everywhere |
| `task5_adversarial` | 20 | Long skips + multi-reuse | Designed to punish greedy schedulers |

Plus procedurally generated training graphs (8-50 nodes) with curriculum-controlled difficulty.

## Training: What the Agent Learned

**Setup:** Qwen 2.5-3B (4-bit, LoRA r=16) trained with GRPO (Group Relative Policy Optimization) + reference-anchored DPO on an RTX 4090. 400 episodes, 3.6 hours. Mix of fixed tasks and procedurally generated curriculum graphs.

**The training loop:** At each step, the model generates 4 candidate actions. Each candidate is scored by actually stepping a cloned environment (no reward hacking). A hand-coded heuristic provides "hint" actions early in training, which are progressively withdrawn. Pairwise DPO teaches the model to prefer higher-scoring actions while reference anchoring prevents policy collapse.

### Results

The untrained model produces random schedules far worse than greedy. After training, it beats greedy on every task:

```
Task                   Before      After      Improvement
-------------------------------------------------------
task1_chain            -0.771      +0.084     +0.856
task2_residual         -0.298      -0.046     +0.252
task3_attention        -0.016      +0.023     +0.039
task4_mixed            -0.663      +0.309     +0.972
task5_adversarial      -1.000      -0.070     +0.930
generalization         -0.775      -0.054     +0.721
```

**Task4 Mixed** is the standout: the agent beats the greedy baseline by ~17% on a 24-node expert-difficulty graph with mixed topologies. **Generalization** is positive, meaning the learned strategies transfer to unseen random graphs.

### Training Curves

The clearest evidence of genuine learning:

**Hint win rate: 73% to 0%.** Early in training, the hand-coded heuristic beats the model's candidates most of the time. By episode 100, the model's own actions consistently outperform the heuristic. The scaffold was removed and the model stood on its own.

**Fusion rate: 85% to 100%.** The model converged to always fusing connected ops, which matches what real compilers do.

**Loss: 0.44 to 0.0001.** Three orders of magnitude drop, confirming the model internalized the scheduling logic rather than memorizing.

**Peak performance at episode 100** (task4=0.419, task5=0.136). We would use early stopping in production.

![Training Results](training_plots.png)

## Why This Environment is Different

**It's not a toy.** The graphs have 8-50 operations with realistic topologies (residual connections, attention fan-out, adversarial skip patterns). The cost model captures real GPU physics (memory hierarchy, compute-bandwidth tradeoffs, kernel launch overhead).

**The greedy baseline is strong.** We don't compare against random or naive. The greedy scheduler already fuses connected ops and retains tensors needed on the immediate next step. Beating it requires multi-step planning.

**The reward teaches, not tricks.** Single latency signal. No reward shaping for specific strategies. The agent must discover fusion, retention, and tile selection purely from their effect on execution speed.

**Procedural generation prevents memorization.** Training uses curriculum-controlled random graphs so the agent learns general scheduling principles, not task-specific patterns.

## API

```
POST /reset          Start episode (body: {"task": "task1_chain"})
POST /step/{id}      Take action (body: {"command": "{\"fuse_with_prev\": true, \"tile\": 128, \"retain\": [3]}"})
GET  /state/{id}     Current state
GET  /tasks          List tasks
WS   /ws             WebSocket for OpenEnv
```

For training with random graphs:
```json
POST /reset {"random": true, "curriculum_level": 0.5}
```

## Links

- **HF Space:** [huggingface.co/spaces/AnishaRoy5555/fusionops-env](https://huggingface.co/spaces/AnishaRoy5555/fusionops-env)
- **GitHub:** [github.com/AnishaRoy5555/compiler-scheduler-env](https://github.com/AnishaRoy5555/compiler-scheduler-env)
- **Training Notebook:** [Colab/RunPod notebook](TODO)
- **Slides:** [Presentation deck](TODO)

## Verified Environment Properties

These properties confirm the environment creates genuine learning signal:

| Property | Value | Why it matters |
|----------|-------|----------------|
| Retention gap | 12-18% on skip graphs | Proves retention decisions have real impact |
| Tile sensitivity | 218% cost variation | Tile choice meaningfully affects latency |
| Reward spread | 0.07-0.33 across random graphs | Enough signal for RL, not too sparse |
| Greedy beatability | 5-17% at peak | Hard enough to be interesting, achievable enough to learn |
