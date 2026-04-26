# First Steps Toward LLM-Guided Compiler Scheduling

*Building an RL environment that trains language models to fuse operations, retain tensors, and configure tiles on GPU computation graphs. What worked, what failed, and what the model actually learned.*

Every ML model compiles to a DAG of tensor operations that must execute on hardware with two tiers of memory: a small fast scratchpad and large slow DRAM. The compiler decides which ops to fuse into kernels, which tensors to keep in fast memory, and how to tile the computation. These three decisions interact, and getting them right can mean a 2-10x performance difference.

We built two RL environments to test whether an LLM could learn these scheduling strategies from scratch. The first, FusionOps, taught us what goes wrong when the environment is too simple. The second, Compiler-Scheduler-Env, produced a model that beats greedy scheduling by up to 17%.

This post covers both environments, the training methodology, and the key findings.

## FusionOps: The Environment That Got Gamed

FusionOps was our first attempt. It had four fixed computation graph tasks (Linear Chain, Diamond Graph, MatMul, K-Split), each with around 6-8 operations. We went through three training iterations on it.

### Run 1: No hints, no learning

We dropped Qwen 2.5-3B into the environment with no prior knowledge. The model received a graph and had to produce scheduling decisions. Result: zero learning. Without any signal about what "fusion" or "retention" meant in this domain, the model could not discover useful strategies through exploration alone.

### Run 2: Scaffolded hints (first breakthrough)

We introduced hints as a strong prior in the reward function, weighted heavily early and progressively discounted as training advanced. The goal was scaffolded learning: guide first, then withdraw.

This worked. The model learned to apply the hints and fuse operations to reduce computation. Scores jumped from 0.0 to 0.3-0.6. This was our first genuine breakthrough.

But retention was never learned. On the Diamond Graph, where a skip connection tensor must be retained across steps, the model scored zero on the retention component.

### Run 3: Two-step lookahead (the insight)

To unlock retention, we introduced a reward function that scored based on immediate reward plus a discounted next-step reward. This was supposed to give the model enough signal to learn when retaining a tensor now would pay off later.

No retention was observed. Instead, we gained a different insight.

The model discovered that fusing all operations was a strategy far superior to selective retention on our small graphs. With only 6-8 operations per task, total fusion was always feasible. A sudden jump in learning toward the final episodes confirmed the model had found this shortcut.

<img width="1335" height="656" alt="v1_ablation" src="https://github.com/user-attachments/assets/2460d787-0911-4295-81a5-927519bac32f" />

*Each training run on FusionOps improved scores. But the mechanism was total fusion, not the intended scheduling strategies.*

<img width="1034" height="654" alt="diamond_proof" src="https://github.com/user-attachments/assets/72a8510c-a479-4407-811d-78f460f67d3d" />

*The Diamond Graph reached 0.553 through total fusion, not selective retention.*

<img width="1485" height="656" alt="v1_tasks" src="https://github.com/user-attachments/assets/f9e48a4a-7c0c-4f83-82f2-0ea967638764" />

*FusionOps scores looked promising across all tasks and runs. The model had found a shortcut, not a strategy.*

### The conclusion

The hints, combined with the environment's inherent structure, gave the model enough signal to game the evaluation rather than solve the underlying problem. The results looked good, but the model had exploited the 8-operation ceiling. OOM errors and partial-fusion tradeoffs only occur in complex computation graphs, and FusionOps was too simple to force the model to learn them.

## Compiler-Scheduler-Env: The Real Challenge

Instead of separate tasks for individual computational graphs, we built a general framework where the environment is agnostic to the complexity or shape of the graph. Unlike FusionOps, which was limited to 8 operations per task, Compiler-Scheduler-Env generates graphs with 8 to 50 nodes across five topology types: linear chains, residual blocks, attention patterns, diamonds, and adversarial configurations with long skip connections.

Two critical design changes made this environment resistant to gaming:

**Greedy baseline instead of naive.** In FusionOps, the baseline was naive (sequential execution, no fusion). In Compiler-Scheduler-Env, the baseline is greedy. The greedy scheduler already performs basic fusion and immediate retention. If the model's score is negative, we know it missed operations that were obvious even to the greedy approach. The model must find strategies the greedy misses.

**Reward: do whatever reduces latency.** Instead of rewarding fusion or retention specifically, we reward any action that reduces latency. This means the model discovers on its own that fusions reduce computation time, that keeping data in fast memory (retention) avoids expensive reloads, and that picking a good tile size matters. The cost model is a physics-based roofline computation: `latency = max(compute, memory) + kernel_launch`. This is the same simulation approach used by TASO, Ansor, and Google's REGAL.

The action space is sequential. The agent walks through nodes in topological order and outputs a JSON action at each step:

```json
{"fuse_with_prev": true, "tile": 128, "retain": [3, 7]}
```

## Training: GRPO with Reference-Anchored DPO

We used GRPO with reference-anchored DPO. In simple terms: generate a few candidate actions, test them all by stepping cloned environments, reward the winners, penalize the losers, and copy the expert when the model is still learning.

Early in training, 50% of candidates come from a hand-coded heuristic. This ratio decays to 10% as the model improves. Reference anchoring (computing loss relative to a frozen base model) prevents policy collapse, which had been a problem with standard PPO in our earlier experiments.

| Parameter | Value |
|-----------|-------|
| Model | Qwen 2.5-3B-Instruct, 4-bit (Unsloth) |
| LoRA | r=16, alpha=32, 29.9M trainable params |
| DPO beta | 0.1 |
| Candidates per step | 4 (model + heuristic, ratio annealing) |
| Curriculum | Difficulty 0.2 to 0.8 |
| Hardware | RTX 4090, RunPod |
| Training time | 3.6h (400 episodes) |

## The Breakthrough

We ran 400 episodes with evaluations at intervals of 50. The findings were clear: this time the LLM was actually able to learn generalized fusion and retention, even on complex compiler operation graphs. This was the problem statement we started with.

```
Task                   Untrained    Trained     Delta
-----------------------------------------------------
task1_chain            -0.771       +0.084      +0.856
task2_residual         -0.298       -0.046      +0.252
task3_attention        -0.016       +0.023      +0.039
task4_mixed            -0.663       +0.309      +0.972
task5_adversarial      -1.000       -0.070      +0.930
generalization         -0.775       -0.054      +0.721
```

<img width="1485" height="656" alt="v2_before_after" src="https://github.com/user-attachments/assets/9c8f1034-de63-440a-9515-d893be825d8d" />

*Red = untrained. Blue = after 400 episodes. Green = best checkpoint during training. Every task improved.*

<img width="1485" height="658" alt="v2_curves" src="https://github.com/user-attachments/assets/7155c865-09e0-41a1-8e3d-f0f1cc101224" />

*Per-task eval scores at each checkpoint. Task4 Mixed (purple) peaks at episode 100 at 0.419.*

## Key Findings

**Learning plateaued after 100 episodes.** Peak performance hit at episode 100 (approximately 50 minutes of training). After that, the learning signals became noisy. The model had matched the heuristic (hint win rate dropped to 0%), and the DPO loss fell to 1e-4 with no more informative pairwise comparisons to drive further learning.

<img width="1335" height="583" alt="reward_curve" src="https://github.com/user-attachments/assets/31ca70cc-59c4-442e-b515-4a9728ab391c" />

*Score stabilizes early. Spikes at episodes 110-170 correspond to curriculum graphs with above-average fusion opportunities.*

<img width="1334" height="583" alt="loss_curve" src="https://github.com/user-attachments/assets/0f065dde-6f05-4f9f-95fe-bbfd2810d64d" />

*Three orders of magnitude loss drop. The signal is exhausted by episode 300.*

**Fusion was learned immediately; retention was not dynamic.** Fusion rate climbed from 85% to 100% within 40 episodes. The model learned "always fuse connected ops when constraints allow," matching production compiler behavior. Retention rate stayed flat at 88% throughout, indicating a fixed policy rather than dynamic adaptation to memory pressure.

<img width="1784" height="581" alt="v2_dynamics" src="https://github.com/user-attachments/assets/3c707cf9-c031-4c47-b963-33694db85ecd" />

*Left: fusion rate converges to 100%. Right: hint win rate drops 73% to 0%. The model surpassed its teacher.*

<img width="1483" height="732" alt="plot5_fusion_retention" src="https://github.com/user-attachments/assets/b02b1e68-a1e1-4ce0-8bc7-97f6b4954f3f" />

*Fusion converges; retention stays flat. One strategy mastered, one not yet.*

**The generalization gap shrank.** FusionOps had a 9.7x gap between fixed-task and generalization scores. Compiler-Scheduler-Env reduced this to 2.3x. Harder training graphs produce more transferable strategies.

<img width="1035" height="659" alt="gen_gap" src="https://github.com/user-attachments/assets/05af4d1b-9d32-4d49-885a-585f27d16b6b" />

*FusionOps to Compiler-Scheduler-Env: generalization gap reduced from 9.7x to 2.3x.*

## Putting It in Context

Right now, compiler scheduling is partly solved by hardcoded rules written by kernel developers, and certain parameters are discovered by users through trial and error. This human intervention typically amounts to around 10-30% optimization.

We achieved 17% optimization in compiler latency, purely through training an LLM in our environment, in our first iteration of Compiler-Scheduler-Env. This was done with a 3B parameter model in 4-bit quantization, trained for under an hour on a single RTX 4090.

The comparison is not direct (simulated cost model vs. hardware profiling, smaller graphs than production), but it demonstrates that LLM-guided scheduling can discover non-trivial optimization strategies through RL.

## Scaling to 8B: Breaking Through the 3B Ceiling

The 3B model on Compiler-Scheduler-Env showed three problems that looked like they could be capacity-related:

**Late-training instability.** The model peaked at episode 100 and then regressed. task5_adversarial went from +0.136 (ep100) to -0.073 (ep400). Generalization swung between -0.108 and +0.060 with no stable trend. The model would find a good policy, then lose it as training continued.

**Generalization plateau at 6%.** The best generalization score was +0.060, meaning the model scheduled unseen graphs only 6% faster than greedy. On fixed tasks it reached +0.419 (task4_mixed), but that knowledge didn't transfer.

**Retention stuck at heuristic level.** Retention rate locked at 88% throughout training, matching the simple rule "retain when future_uses > 0". The model never learned conditional retention.

We scaled to Llama-3.1-8B to find out whether these were model capacity limits or environment limits.

### What changed

| | Qwen2.5-3B (env2_v1) | Llama-3.1-8B (env2_v2) |
|---|---|---|
| Parameters | 3B (4-bit QLoRA) | 8B (4-bit QLoRA) |
| LoRA | r=16, alpha=32 | r=32, alpha=64 |
| Learning rate | 1e-5 | 5e-6 (halved for stability) |
| Grad accumulation | 4 | 8 (doubled for smoother updates) |
| Episodes | 400 | 400 |
| GPU | RTX 4090 (24GB) | RTX 5090 (32GB) |
| Training time | 3.6h | 2.9h |

### Results

```
Task                   3B Final    8B Final    Change
-----------------------------------------------------
task1_chain            +0.084      +0.084      Same (ceiling)
task2_residual         -0.046      +0.015      +0.061
task3_attention        +0.023      +0.034      +0.011
task4_mixed            +0.309      +0.573      +85%
task5_adversarial      -0.070      +0.131      Flipped to positive
generalization         -0.054      +0.252      Flipped to positive
```

### Problem 1: Late-training instability - SOLVED

The 3B model's best scores all occurred at episode 100, then decayed. The 8B model improved continuously through all 400 episodes with no regression:

| Checkpoint | 3B generalization | 8B generalization |
|---|---|---|
| ep 50 | -0.138 | -0.298 |
| ep 100 | 0.044 | 0.032 |
| ep 150 | 0.059 | 0.081 |
| ep 200 | 0.009 | 0.088 |
| ep 250 | -0.108 | 0.186 |
| ep 300 | 0.060 | 0.088 |
| ep 350 | 0.037 | 0.175 |
| ep 400 | -0.020 | 0.258 |

The 3B model oscillated between -0.108 and +0.060 across checkpoints, never finding a stable policy. The 8B model climbed steadily and was still rising at ep400, suggesting more episodes would push generalization even higher.

### Problem 2: Generalization plateau - SOLVED

The 3B model achieved 6% average improvement on unseen graphs at best. The 8B model achieved 25%. A 4x improvement in the metric that matters most.

The gap between fixed-task and generalization scores narrowed from 7x (3B: 0.419/0.060) to 2.3x (8B: 0.573/0.252). The 8B model learned general scheduling principles that transfer to arbitrary graph topologies, not just task-specific tricks.

### Problem 3: Retention plateau - NOT SOLVED

Retention rate stayed at 88% for both models, matching the simple heuristic "retain when future_uses > 0". Neither model learned conditional retention. This is now confirmed as a limitation of the reward signal and environment design, not model capacity. The 2-step lookahead makes immediate retention visible but does not capture the value of retaining a tensor for a consumer 5+ steps away.

### What this means in scheduling terms

The greedy baseline represents a basic heuristic scheduler that already does simple fusion and immediate retention, roughly equivalent to a first-pass compiler heuristic.

**3B result (previous):** 42% faster than greedy on the best fixed task, 6% on unseen graphs. Averaged across all tasks, approximately 17% scheduling improvement.

**8B result:** 57% faster than greedy on the hardest fixed task (24 ops, mixed topology), and 25% faster on graphs the model has never seen. The generalization number is the one that matters for real deployment: the model has learned general scheduling principles that transfer to arbitrary computation graph topologies.

<img width="1485" height="658" alt="v2_curves" src="https://github.com/user-attachments/assets/d9720cf1-96a7-4655-95fa-0e239fd99436" />

*8B model learning curves. task4_mixed (purple) climbs steadily to 0.571. Generalization (blue) reaches 0.258 and is still rising.*

## What Comes Next

**Multi-turn context.** Each scheduling step is currently stateless. Adding the last 3-4 actions as conversation history should improve retention on long-skip graphs, where the model needs to reason about what it retained several steps ago.

**Larger models.** A 7B or 14B model may acquire selective retention strategies that 3B cannot represent.

**Self-play.** Once the model matches the heuristic, learning stalls. Dropping the heuristic and scoring candidates only against each other removes the ceiling.

**Transfer validation.** Validating whether learned strategies transfer to real compiler backends (XLA, Triton) is the most important open question.

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
- - **env2_v2** - [colab.research.google.com/drive/14vvCB8DENaqtoGKViaO2gjXtUmO-lAWD?usp=sharing(https://colab.research.google.com/drive/14vvCB8DENaqtoGKViaO2gjXtUmO-lAWD?usp=sharing)<br><br>
- **Presentation:** [Slides](https://drive.google.com/drive/folders/1vVotqQQYkDCo3FoNRwuzQyg8LG4_NxHd?usp=sharing)

---

*Anisha Roy and Anshul Badhani. OpenEnv Hackathon India 2026.*
