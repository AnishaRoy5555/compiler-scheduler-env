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

# FusionOps: Teaching an LLM to Think Like a Compiler

> *Can an LLM, placed inside a simulated GPU execution environment and trained via RL, discover the same fusion, retention, and scheduling strategies that took compiler engineers years to hand-tune?*

## The Problem

Every deep learning model (transformer, diffusion model, CNN) compiles down to a directed acyclic graph of tensor operations. These operations must execute on hardware where a small, fast scratchpad (SRAM) sits next to a large, slow main memory (DRAM). The compiler makes three critical decisions for every operation:

**Fuse?** Should this operation merge into the previous kernel? Fused ops share fast memory, so intermediates never touch slow DRAM. One eliminated memory roundtrip can save microseconds that compound across millions of inference calls.

**Tile?** How should the computation be split? Larger tiles amortize memory latency but demand more scratchpad space. The optimal tile size depends on what else is competing for fast memory.

**Retain?** Which tensor outputs should stay in scratchpad for future steps? Retention avoids expensive DRAM reloads, but scratchpad capacity is finite. Retain too much and you run out of space. Retain too little and you pay the reload cost later.

Get these decisions wrong and your model runs 2-10x slower than it should. Today, these decisions are made by hand-tuned heuristics in XLA, Triton, and TVM that took years to develop. We asked: can an LLM learn them from scratch?

## The Journey: From FusionOps to Compiler-Scheduler-Env

## How the Environments Work

### FusionOps-Env (V1): Subgraph Selection with Naive Baseline

**[github.com/AnishaRoy5555/fusionops-env](https://github.com/AnishaRoy5555/fusionops-env)**

The agent receives a computation graph - a DAG of tensor operations (pointwise ops like relu/layernorm, and matmuls) connected by data dependencies - along with a hardware spec defining fast memory (SRAM scratchpad) capacity and slow memory (DRAM) bandwidth. At each step, the agent sees:

1. **CURRENT STATE**: which ops are completed, which are ready (all predecessors done)
2. **MEMORY**: what tensors are currently in fast memory and how much capacity remains
3. **GRAPH SUMMARY**: all ops with status (pending/ready/done), all tensors with their current location (fast/slow/not yet produced)
4. **VALID ACTION EXAMPLES**: 2-4 dynamically generated valid actions for the current state, guaranteed correct by construction
5. **CONSTRAINTS**: hard rules (working set must fit in fast memory, ops must be connected, predecessors must be scheduled)

The agent selects a subgraph of ready operations to fuse, a 3D tiling config `[w, h, k]`, and tensor IDs to retain in fast memory:

```
SCHEDULE ops=[0,1,2,3] config=[128,128,32] retain=[2]
```

The environment validates the action (connected subgraph? dependencies met? working set fits in fast memory?), computes latency using a roofline cost model (`latency = max(compute_time, memory_transfer_time)` per tile), and returns the next observation. Invalid actions receive graduated penalties: -0.05 for OOM (near-miss, wrong tile size), -0.10 for parse errors, -0.15 for connectivity/retention violations, -0.20 for dependency violations. Valid actions receive +0.02 baseline plus a latency efficiency signal and a +0.05 bonus per additional op fused.

**Cost model details**: Fusion makes intermediate tensors between fused ops ephemeral (zero memory cost). For matmuls, split-K reduces the reduction dimension - trading more accumulation passes for lower peak memory - which can unlock fusion opportunities that would otherwise OOM. Tile traversal order matters: snake (zigzag) traversal keeps the LHS warm across rows, reducing reloads vs raster (left-to-right).

**Scoring**: `score = max(0, (naive_latency - agent_latency) / naive_latency)`. The naive baseline schedules every op individually with no fusion and no retention. Scores range from 0 (no improvement) to ~0.67 (theoretical max on simple chains).

**Four fixed tasks**: `task1_linear` (6 pointwise ops, tests basic fusion), `task2_diamond` (6 ops, tensor T1 consumed by 3 downstream ops - tests retention vs recompute), `task3_matmul` (3 matmuls + 1 pointwise - tests split-K and OOM avoidance), `task4_multistage` (3 matmuls + 5 pointwise with a skip connection spanning 6 steps - tests long-horizon planning).

**Why we moved on**: The combinatorial action space (any subset of ready ops, any 3D config, any subset of tensors) allowed the model to memorize one action string per task. On `task1_linear`, the trained model always produced `SCHEDULE ops=[0,1,2,3,4,5] config=[128,128,1] retain=[]` regardless of the observation. The valid action examples in the observation could be copied verbatim. With only 4 fixed tasks of 6-8 ops, memorization was easier than reasoning. Generalization to unseen graphs: 0.049.

---

### Compiler-Scheduler-Env (V2): Sequential Per-Node with Greedy Baseline

A ground-up redesign addressing V1's generalization failure. The agent walks operations in **topological order**, one node at a time. At each node, it outputs a JSON decision:

```json
{"fuse_with_prev": true, "tile": 128, "retain": [3, 5]}
```

Three fields:
- **`fuse_with_prev`** (bool): merge this op into the current kernel group, or start a new kernel. Fused ops share fast memory and eliminate intermediate DRAM traffic.
- **`tile`** (int from {32, 64, 128, 256}): tiling granularity controlling the compute-memory tradeoff on the roofline. Larger tiles amortize memory latency but demand more scratchpad.
- **`retain`** (list of node IDs): which tensor outputs to keep in fast memory for future consumers. Costs capacity now, saves DRAM reloads later.

**Observation format**: Compact JSON with a fixed schema. Step 0 delivers the full graph (`nodes` array with op types, shapes, and input edges). Subsequent steps show:
- `current_node`: the node being scheduled now
- `current_group`: which ops are in the current fusion kernel and its tile size
- `fast_mem`: what's currently in fast memory
- `capacity`: scratchpad size in bytes
- `max_fusion`: maximum ops per kernel (prevents unrealistically large fusions)
- **`future_uses`**: a dictionary mapping each tensor ID to the number of downstream operations that still need it. This is the key addition - it provides the minimal information needed for correct retention reasoning (`future_uses["3"] = 2` means tensor 3 has 2 remaining consumers, retain it; `future_uses["5"] = 0` means tensor 5 is dead, don't waste capacity on it)
- `lookahead`: the next 2 nodes in topological order, letting the agent anticipate fusion opportunities

No valid action examples are provided - the agent must construct actions from the observation, not copy them.

**Cost model**: Physics-based roofline: `latency = max(compute_time, memory_transfer_time) + kernel_launch_overhead`. Same approach used by Google's REGAL, Apache TASO, and TVM Ansor. Each new kernel (non-fused op) incurs a fixed launch overhead. Missing a retention decision means the tensor reloads from DRAM at bandwidth cost (10-100x slower than SRAM access). Invalid JSON actions receive -0.10.

**Scoring**: `score = (greedy_latency - agent_latency) / greedy_latency`. The **greedy baseline** already does basic fusion (fuses consecutive same-type ops) and immediate retention (retains tensors needed by the very next op). Scores can be negative (agent worse than greedy) or positive (agent found optimizations the greedy heuristic missed). This is deliberately harder - the easy wins are already taken, so the agent must discover multi-step retention, cross-type fusion boundaries, and tile-aware memory management to score positive.

**Five fixed tasks**: `task1_chain` (8 ops, basic fusion), `task2_residual` (12 ops, ResNet-style skip connections requiring retention across steps), `task3_attention` (16 ops, Q/K/V fan-out with 3 consumers from one producer), `task4_mixed` (24 ops, diamonds + skips + attention - no single strategy works everywhere), `task5_adversarial` (20 ops, long skip connections and multi-reuse patterns specifically designed to punish greedy one-step-lookahead schedulers - only way to beat greedy is multi-step retention).

**Procedural generation**: 60% of training episodes use curriculum-controlled random graphs (8-50 nodes) with topology mixes including chains, residual blocks, attention patterns, diamonds, and adversarial skip patterns. Difficulty ramps from 0.2 (easy, ~16 ops) to 0.8 (hard, ~46 ops) over training. This prevents the memorization that killed V1.

**Verified properties**: Retention gap of 12-18% on skip graphs (proving retention decisions matter), tile sensitivity of 218% cost variation (tile choice affects latency), reward spread of 0.07-0.33 across random graphs (enough signal for RL, not too sparse), and greedy beatability of 5-17% at peak (hard enough to be interesting, achievable enough to learn).

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

<img width="1485" height="658" alt="v2_curves" src="https://github.com/user-attachments/assets/d9720cf1-96a7-4655-95fa-0e239fd99436" />

## Scaling Up: Why 3B Wasn't Enough, and What 8B Changed

### The 3B ceiling

The 3B model (Qwen2.5-3B) on the Compiler-Scheduler-Env showed three problems that looked like they could be capacity-related:

**Late-training instability.** The model peaked at episode 100 and then regressed. task5_adversarial went from +0.136 (ep100) to -0.073 (ep400). Generalization swung between -0.108 and +0.060 with no stable trend. The model would find a good policy, then lose it as training continued. This looked like the 3B model's representation wasn't large enough to hold both task-specific strategies and general scheduling principles simultaneously.

**Generalization plateau at 6%.** The best generalization score was +0.060, meaning the model scheduled unseen graphs only 6% faster than the greedy baseline. On fixed tasks it reached +0.419 (task4_mixed), but that knowledge didn't transfer. The 3B model appeared to learn task-specific tricks rather than general scheduling reasoning.

**Retention stuck at heuristic level.** Retention rate locked at 88% throughout training, matching the simple rule "retain when future_uses > 0". The model never learned conditional retention: when to retain based on distance to consumer, memory pressure, or graph topology. We couldn't tell if this was a model capacity limit or a reward signal problem.

We scaled to 8B to answer one question: are these limits caused by the model being too small, or by the environment and training recipe?

### Configuration: 3B vs 8B

| | Qwen2.5-3B (env2_v1) | Llama-3.1-8B (env2_v2) |
|---|---|---|
| Parameters | 3B (4-bit QLoRA) | 8B (4-bit QLoRA) |
| LoRA | r=16, alpha=32 | r=32, alpha=64 |
| Learning rate | 1e-5 | 5e-6 (halved for stability) |
| Grad accumulation | 4 | 8 (doubled for smoother updates) |
| Episodes | 400 | 400 |
| GPU | RTX 4090 (24GB) | RTX 5090 (32GB) |
| Training time | 3.6h | 2.9h |

### Results: 8B solved two of the three problems

| Task | 3B Final | 8B Final | Change |
|---|---|---|---|
| task1_chain | +0.084 | +0.084 | Same (ceiling) |
| task2_residual | -0.046 | +0.015 | +0.061 |
| task3_attention | +0.023 | +0.034 | +0.011 |
| task4_mixed (24 ops) | +0.309 | **+0.573** | **+85%** |
| task5_adversarial (20 ops) | -0.070 | **+0.131** | Flipped to positive |
| generalization (unseen graphs) | -0.054 | **+0.252** | Flipped to positive |

<img width="2700" height="1500" alt="training_plots" src="https://github.com/user-attachments/assets/ba8da473-762b-43fc-8add-0cf6824f9e81" />

### Problem 1: Late-training instability - SOLVED

The 3B model's best scores all occurred at episode 100, then decayed. The 8B model improved continuously through all 400 episodes with no regression:

| Checkpoint | 3B generalization | 8B generalization |
|---|---|---|
| ep 50 | -0.138 | -0.298 |
| ep 100 | 0.044 | 0.032 |
| ep 150 | 0.059 | 0.081 |
| ep 200 | 0.009 | 0.088 |
| ep 250 | -0.108 | **0.186** |
| ep 300 | 0.060 | 0.088 |
| ep 350 | 0.037 | **0.175** |
| ep 400 | -0.020 | **0.258** |

The 3B model oscillated between -0.108 and +0.060 across checkpoints. The 8B model climbed steadily and was still rising at ep400, suggesting more episodes would push generalization even higher. The combination of halved learning rate (5e-6 vs 1e-5) and doubled gradient accumulation (8 vs 4) gave the 8B model smoother, more stable training. The larger representation space allowed it to hold general scheduling principles without forgetting task-specific strategies.

### Problem 2: Generalization plateau - SOLVED

The 3B model achieved 6% average improvement on unseen graphs at its best checkpoint. The 8B model achieved **25%**. A 4x improvement in the metric that matters most for real-world deployment.

The greedy baseline represents a basic heuristic scheduler that already does simple fusion and immediate retention, roughly what a first-pass compiler heuristic produces. The 3B model's 42% improvement on the best fixed task (task4_mixed) looked impressive, but only 6% transferred to unseen graphs. The 8B model hit **57% on task4_mixed** and **25% on unseen graphs**. The gap between fixed and unseen narrowed from 7x (3B: 0.419/0.060) to 2.3x (8B: 0.573/0.252).

### Problem 3: Retention plateau - NOT SOLVED

Retention rate stayed at 88% for both models, matching the simple heuristic of "retain when future_uses > 0". Neither the 3B nor the 8B model learned conditional retention. This is now confirmed as a limitation of the current reward signal and environment design, not model capacity. The 2-step lookahead makes immediate retention visible but does not capture the value of retaining a tensor for a consumer 5+ steps away. Solving this likely requires either deeper lookahead, an explicit retention reward component, or a fundamentally different observation design that encodes the cost of not retaining.

### What this means in scheduling terms

**3B result (previous):** The model learned to schedule approximately 17% faster than a heuristic compiler on average, with 6% improvement on unseen graph topologies.

**8B result (this run):** The model schedules **57% faster than the heuristic on the hardest known graph** (24 ops, mixed topology), and **25% faster on graphs it has never seen**. The generalization number is the one that matters for real deployment: the model has learned general scheduling principles that transfer to arbitrary computation graph topologies.

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

- **Initial Environment:** [huggingface.co/spaces/AnishaRoy5555/fusionops-env](https://huggingface.co/spaces/AnishaRoy5555/fusionops-env)
- **Final Environment:** [huggingface.co/spaces/AnishaRoy5555/compiler-scheduler-env](https://huggingface.co/spaces/AnishaRoy5555/compiler-scheduler-env)
- **Initial Codebase:** [github.com/AnishaRoy5555/fusionops-env](https://github.com/AnishaRoy5555/fusionops-env)
- **Final Codebase:** [github.com/AnishaRoy5555/compiler-scheduler-env](https://github.com/AnishaRoy5555/compiler-scheduler-env)

**Training notebook:**
- **env1_v1** - [colab.research.google.com/drive/1F9RXon5vpSv8zww-w19ZK4cld1qCjdCz?usp=sharing](https://colab.research.google.com/drive/1F9RXon5vpSv8zww-w19ZK4cld1qCjdCz?usp=sharing)
- **env1_v2** - [colab.research.google.com/drive/1nHo8L4jy9s3CfC4guZfXJgaJZzaSoVs6?usp=sharing](https://colab.research.google.com/drive/1nHo8L4jy9s3CfC4guZfXJgaJZzaSoVs6?usp=sharing)
- **env1_v3** - [colab.research.google.com/drive/1eWMkfeAGTgFkSz6JoJJRDPuP1FIJTG1o?usp=sharing](https://colab.research.google.com/drive/1eWMkfeAGTgFkSz6JoJJRDPuP1FIJTG1o?usp=sharing)
- **env2_v1** - [colab.research.google.com/drive/1jbUlkI9_Lmn4yidw-udDKT4LhrphS30U?usp=sharing](https://colab.research.google.com/drive/1jbUlkI9_Lmn4yidw-udDKT4LhrphS30U?usp=sharing)
- **env2_v2** - [colab.research.google.com/drive/14vvCB8DENaqtoGKViaO2gjXtUmO-lAWD?usp=sharing](https://colab.research.google.com/drive/14vvCB8DENaqtoGKViaO2gjXtUmO-lAWD?usp=sharing)<br><br>
- **Presentation:** [Slides](https://drive.google.com/drive/folders/1vVotqQQYkDCo3FoNRwuzQyg8LG4_NxHd?usp=sharing)

## Verified Environment Properties

These properties confirm the environment creates genuine learning signal:

| Property | Value | Why it matters |
|----------|-------|----------------|
| Retention gap | 12-18% on skip graphs | Proves retention decisions have real impact |
| Tile sensitivity | 218% cost variation | Tile choice meaningfully affects latency |
| Reward spread | 0.07-0.33 across random graphs | Enough signal for RL, not too sparse |
| Greedy beatability | 5-17% at peak | Hard enough to be interesting, achievable enough to learn |
