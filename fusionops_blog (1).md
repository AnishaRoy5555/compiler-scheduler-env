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

![V1 ablation](v1_ablation.png)
*Each training run on FusionOps improved scores. But the mechanism was total fusion, not the intended scheduling strategies.*

![Diamond proof](diamond_proof.png)
*The Diamond Graph reached 0.553 through total fusion, not selective retention.*

![V1 per-task scores](v1_tasks.png)
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

![Before vs after](v2_before_after.png)
*Red = untrained. Blue = after 400 episodes. Green = best checkpoint during training. Every task improved.*

![Learning curves](v2_curves.png)
*Per-task eval scores at each checkpoint. Task4 Mixed (purple) peaks at episode 100 at 0.419.*

## Key Findings

**Learning plateaued after 100 episodes.** Peak performance hit at episode 100 (approximately 50 minutes of training). After that, the learning signals became noisy. The model had matched the heuristic (hint win rate dropped to 0%), and the DPO loss fell to 1e-4 with no more informative pairwise comparisons to drive further learning.

![Reward trajectory](reward_curve.png)
*Score stabilizes early. Spikes at episodes 110-170 correspond to curriculum graphs with above-average fusion opportunities.*

![Loss convergence](loss_curve.png)
*Three orders of magnitude loss drop. The signal is exhausted by episode 300.*

**Fusion was learned immediately; retention was not dynamic.** Fusion rate climbed from 85% to 100% within 40 episodes. The model learned "always fuse connected ops when constraints allow," matching production compiler behavior. Retention rate stayed flat at 88% throughout, indicating a fixed policy rather than dynamic adaptation to memory pressure.

![Fusion and hint win dynamics](v2_dynamics.png)
*Left: fusion rate converges to 100%. Right: hint win rate drops 73% to 0%. The model surpassed its teacher.*

![Fusion and retention detail](plot5_fusion_retention.png)
*Fusion converges; retention stays flat. One strategy mastered, one not yet.*

**The generalization gap shrank.** FusionOps had a 9.7x gap between fixed-task and generalization scores. Compiler-Scheduler-Env reduced this to 2.3x. Harder training graphs produce more transferable strategies.

![Generalization gap](gen_gap.png)
*FusionOps to Compiler-Scheduler-Env: generalization gap reduced from 9.7x to 2.3x.*

## Putting It in Context

Right now, compiler scheduling is partly solved by hardcoded rules written by kernel developers, and certain parameters are discovered by users through trial and error. This human intervention typically amounts to around 10-30% optimization.

We achieved 17% optimization in compiler latency, purely through training an LLM in our environment, in our first iteration of Compiler-Scheduler-Env. This was done with a 3B parameter model in 4-bit quantization, trained for under an hour on a single RTX 4090.

The comparison is not direct (simulated cost model vs. hardware profiling, smaller graphs than production), but it demonstrates that LLM-guided scheduling can discover non-trivial optimization strategies through RL.

## What Comes Next

**Multi-turn context.** Each scheduling step is currently stateless. Adding the last 3-4 actions as conversation history should improve retention on long-skip graphs, where the model needs to reason about what it retained several steps ago.

**Larger models.** A 7B or 14B model may acquire selective retention strategies that 3B cannot represent.

**Self-play.** Once the model matches the heuristic, learning stalls. Dropping the heuristic and scoring candidates only against each other removes the ceiling.

**Transfer validation.** Validating whether learned strategies transfer to real compiler backends (XLA, Triton) is the most important open question.

## Links

- **Environment:** [huggingface.co/spaces/AnishaRoy5555/fusionops-env](https://huggingface.co/spaces/AnishaRoy5555/fusionops-env)
- **Code:** [github.com/AnishaRoy5555/compiler-scheduler-env](https://github.com/AnishaRoy5555/compiler-scheduler-env)
- **Training notebook:** [RunPod](TODO)
- **Presentation:** [Slides](TODO)

---

*Anisha Roy and Anshul Chauhan. OpenEnv Hackathon India 2026.*
