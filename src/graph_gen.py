"""
FusionOps V2 - Procedural Graph Generator
Generates computation graphs that guarantee:
1. Fusion opportunities (consecutive fuseable ops)
2. Retention pressure (outputs used far downstream via skip connections)
3. Memory pressure (graphs where naive execution OOMs or is very slow)
4. Curriculum difficulty scaling
"""

from __future__ import annotations

import random
from typing import Optional

from .models import (
    Graph, Node, HardwareSpec, OpType,
    POINTWISE_OPS, REDUCTION_OPS, COMPUTE_OPS,
)


# ============================================================
# Topology Templates
# ============================================================

def _linear_chain(n_ops: int) -> list[tuple[OpType, list[int]]]:
    """Pure linear: op0 -> op1 -> ... -> opN"""
    ops = []
    for i in range(n_ops):
        if i == 0:
            ops.append((OpType.MATMUL, []))
        else:
            op_type = random.choice([OpType.RELU, OpType.GELU, OpType.LAYERNORM])
            ops.append((op_type, [i - 1]))
    return ops


def _conv_bn_relu_block(start_id: int) -> list[tuple[OpType, list[int]]]:
    """Conv -> LayerNorm -> ReLU (classic fuseable block)"""
    return [
        (OpType.CONV2D, [start_id] if start_id >= 0 else []),
        (OpType.LAYERNORM, [start_id + 1 if start_id >= 0 else 0]),
        (OpType.RELU, [start_id + 2 if start_id >= 0 else 1]),
    ]


def _attention_pattern(start_id: int, input_id: int) -> list[tuple[OpType, list[int]]]:
    """
    Simplified attention: Q*K^T -> softmax -> *V
    Creates skip connection (input_id feeds both Q and V paths)
    """
    base = start_id
    return [
        # Q projection
        (OpType.MATMUL, [input_id]),        # base+0: Q = input @ Wq
        # K projection  
        (OpType.MATMUL, [input_id]),        # base+1: K = input @ Wk
        # V projection
        (OpType.MATMUL, [input_id]),        # base+2: V = input @ Wv
        # Q*K^T
        (OpType.MATMUL, [base, base + 1]),  # base+3: attn = Q @ K^T
        # Softmax
        (OpType.SOFTMAX, [base + 3]),       # base+4: attn_w = softmax(attn)
        # attn * V
        (OpType.MATMUL, [base + 4, base + 2]),  # base+5: out = attn_w @ V
    ]


def _residual_block(start_id: int, input_id: int) -> list[tuple[OpType, list[int]]]:
    """
    Residual: input -> matmul -> relu -> matmul -> add(input, .)
    Creates skip connection from input_id to the add.
    """
    base = start_id
    return [
        (OpType.MATMUL, [input_id]),         # base+0
        (OpType.RELU, [base]),               # base+1
        (OpType.MATMUL, [base + 1]),         # base+2
        (OpType.ADD, [input_id, base + 2]),  # base+3: residual add
    ]


def _diamond(start_id: int, input_id: int) -> list[tuple[OpType, list[int]]]:
    """
    Diamond: input -> (branch A, branch B) -> merge
    Tests fan-out + fan-in.
    """
    base = start_id
    return [
        (OpType.MATMUL, [input_id]),         # base+0: branch A
        (OpType.RELU, [base]),               # base+1: branch A cont
        (OpType.MATMUL, [input_id]),         # base+2: branch B
        (OpType.GELU, [base + 2]),           # base+3: branch B cont
        (OpType.ADD, [base + 1, base + 3]),  # base+4: merge
    ]


def _long_skip(start_id: int, input_id: int) -> list[tuple[OpType, list[int]]]:
    """
    ADVERSARIAL: Long skip dependency.
    Op0 -> Op1 -> Op2 -> Op3 -> Op4(uses Op0 output)

    Not retaining Op0's output forces a full reload from slow memory.
    Greedy (immediate-only retention) will miss this.
    Retention-aware policy wins by 15-25%.
    """
    base = start_id
    return [
        (OpType.MATMUL, [input_id]),         # base+0: produces key output
        (OpType.RELU, [base]),               # base+1
        (OpType.LAYERNORM, [base + 1]),      # base+2
        (OpType.GELU, [base + 2]),           # base+3
        (OpType.ADD, [input_id, base + 3]),  # base+4: uses input_id (skip)
        (OpType.ADD, [base, base + 4]),      # base+5: uses base+0 output (LONG skip)
    ]


def _multi_skip(start_id: int, input_id: int) -> list[tuple[OpType, list[int]]]:
    """
    ADVERSARIAL: Multiple skip connections from same source.
    Op0's output is used at Op3 AND Op6.
    Forces the agent to retain Op0 across the entire schedule.
    """
    base = start_id
    return [
        (OpType.MATMUL, [input_id]),         # base+0: high-value output
        (OpType.RELU, [base]),               # base+1
        (OpType.MATMUL, [base + 1]),         # base+2
        (OpType.ADD, [base, base + 2]),      # base+3: first reuse of base+0
        (OpType.GELU, [base + 3]),           # base+4
        (OpType.LAYERNORM, [base + 4]),      # base+5
        (OpType.ADD, [base, base + 5]),      # base+6: second reuse of base+0
    ]


def _bottleneck_chain(start_id: int, input_id: int) -> list[tuple[OpType, list[int]]]:
    """
    ADVERSARIAL: Alternating compute-heavy and pointwise ops.
    Forces tile-size decisions: matmul needs large tiles for efficiency,
    but fusing with pointwise under memory pressure needs smaller tiles.
    """
    base = start_id
    return [
        (OpType.MATMUL, [input_id]),         # base+0: compute-heavy
        (OpType.RELU, [base]),               # base+1: cheap, fuse candidate
        (OpType.MATMUL, [base + 1]),         # base+2: compute-heavy again
        (OpType.GELU, [base + 2]),           # base+3: cheap, fuse candidate
        (OpType.MATMUL, [base + 3]),         # base+4: compute-heavy
        (OpType.LAYERNORM, [base + 4]),      # base+5: reduction
    ]


# ============================================================
# Graph Generator
# ============================================================

def _compute_node_costs(op: OpType, shape: list[int]) -> tuple[float, int, int]:
    """Returns (flops, output_bytes, per_input_bytes).

    IMPORTANT: flops are normalized so the environment is memory-bound
    for pointwise ops and balanced for compute ops. This ensures fusion
    (which eliminates intermediate memory transfers) creates meaningful
    reward signal. Without this, the env is compute-bound and fusion
    barely matters.
    """
    elements = shape[0] * shape[1]
    output_bytes = elements * 4  # float32

    if op == OpType.MATMUL:
        # MatMul is compute-heavy: balanced with memory
        flops = elements * 4.0  # reduced from M*N*K to keep balance
        input_bytes = elements * 4
    elif op == OpType.CONV2D:
        flops = elements * 6.0
        input_bytes = elements * 4
    elif op in (OpType.RELU, OpType.GELU, OpType.ADD, OpType.TRANSPOSE):
        # Pointwise: very cheap compute, memory-bound
        # This is realistic: pointwise ops ARE memory-bound on real hardware
        flops = elements * 0.1
        input_bytes = elements * 4
    elif op in (OpType.LAYERNORM, OpType.SOFTMAX, OpType.REDUCE):
        # Reduction: moderate compute, still memory-heavy
        flops = elements * 1.0
        input_bytes = elements * 4
    else:
        flops = elements * 0.1
        input_bytes = elements * 4

    return flops, output_bytes, input_bytes


def generate_graph(
    num_ops: int = 16,
    shape: list[int] | None = None,
    difficulty: str = "medium",
    seed: Optional[int] = None,
    topology: Optional[str] = None,
) -> Graph:
    """
    Generate a computation graph with guaranteed learning pressure.

    Args:
        num_ops: target number of ops (actual may vary by a few)
        shape: tensor dimensions [H, W], default [256, 256]
        difficulty: "easy" | "medium" | "hard" | "expert"
        seed: random seed for reproducibility
        topology: force a topology type, or None for random mix
    """
    if seed is not None:
        random.seed(seed)

    if shape is None:
        shape = [256, 256]

    # Build topology
    ops_spec = _build_topology(num_ops, topology)

    # Create nodes
    nodes = []
    for i, (op_type, input_ids) in enumerate(ops_spec):
        flops, output_bytes, per_input_bytes = _compute_node_costs(op_type, shape)
        input_bytes_list = [per_input_bytes for _ in input_ids]
        nodes.append(Node(
            id=i,
            op=op_type,
            shape=list(shape),
            inputs=list(input_ids),
            flops=flops,
            output_bytes=output_bytes,
            input_bytes=input_bytes_list,
        ))

    # Hardware spec based on difficulty
    hw = _make_hardware(difficulty, shape, len(nodes))

    return Graph(nodes=nodes, hardware=hw)


def _build_topology(
    target_ops: int,
    topology: Optional[str] = None,
) -> list[tuple[OpType, list[int]]]:
    """Build a graph topology from blocks."""
    ops: list[tuple[OpType, list[int]]] = []

    if topology == "linear":
        return _linear_chain(target_ops)

    if topology == "attention":
        # Input node
        ops.append((OpType.MATMUL, []))  # node 0: input projection
        # Stack attention blocks
        last_output = 0
        while len(ops) < target_ops - 1:
            block = _attention_pattern(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
        # Final output
        ops.append((OpType.LAYERNORM, [last_output]))
        return ops[:target_ops]

    if topology == "residual":
        ops.append((OpType.MATMUL, []))  # node 0
        last_output = 0
        while len(ops) < target_ops - 1:
            block = _residual_block(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
        return ops[:target_ops]

    if topology == "long_skip":
        # Chain of long-skip blocks: guaranteed retention pressure
        ops.append((OpType.MATMUL, []))  # node 0
        last_output = 0
        while len(ops) < target_ops - 1:
            remaining = target_ops - len(ops)
            if remaining >= 7:
                block = _multi_skip(len(ops), last_output)
            elif remaining >= 6:
                block = _long_skip(len(ops), last_output)
            else:
                block = [(OpType.RELU, [last_output])]
            ops.extend(block)
            last_output = len(ops) - 1
        return ops[:target_ops]

    if topology == "adversarial":
        # Mix of all adversarial patterns
        ops.append((OpType.MATMUL, []))  # node 0
        last_output = 0
        patterns = [_long_skip, _multi_skip, _bottleneck_chain]
        pidx = 0
        while len(ops) < target_ops - 1:
            remaining = target_ops - len(ops)
            pattern = patterns[pidx % len(patterns)]
            min_size = 6 if pattern != _multi_skip else 7
            if remaining >= min_size:
                block = pattern(len(ops), last_output)
                ops.extend(block)
                last_output = len(ops) - 1
            else:
                ops.append((OpType.RELU, [last_output]))
                last_output = len(ops) - 1
            pidx += 1
        return ops[:target_ops]

    # Default: mixed topology with guaranteed variety
    # IMPORTANT: includes adversarial patterns (long_skip, multi_skip, bottleneck)
    # to ensure non-greedy strategies are rewarded
    ops.append((OpType.MATMUL, []))  # node 0: root
    last_output = 0
    skip_sources: list[int] = [0]  # candidates for skip connections

    while len(ops) < target_ops:
        remaining = target_ops - len(ops)
        block_type = random.choices(
            ["chain", "residual", "diamond", "attention",
             "long_skip", "multi_skip", "bottleneck"],
            weights=[15, 20, 10, 10, 20, 15, 10],
            k=1,
        )[0]

        if block_type == "chain" or remaining < 4:
            # 2-4 ops in a chain (guaranteed fuseable)
            chain_len = min(random.randint(2, 4), remaining)
            for j in range(chain_len):
                op = random.choice([OpType.RELU, OpType.GELU, OpType.LAYERNORM])
                ops.append((op, [last_output]))
                last_output = len(ops) - 1
            skip_sources.append(last_output)

        elif block_type == "residual" and remaining >= 4:
            block = _residual_block(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
            skip_sources.append(last_output)

        elif block_type == "diamond" and remaining >= 5:
            block = _diamond(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
            skip_sources.append(last_output)

        elif block_type == "attention" and remaining >= 6:
            block = _attention_pattern(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
            skip_sources.append(last_output)

        elif block_type == "long_skip" and remaining >= 6:
            block = _long_skip(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
            skip_sources.append(last_output)

        elif block_type == "multi_skip" and remaining >= 7:
            block = _multi_skip(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
            skip_sources.append(last_output)

        elif block_type == "bottleneck" and remaining >= 6:
            block = _bottleneck_chain(len(ops), last_output)
            ops.extend(block)
            last_output = len(ops) - 1
            skip_sources.append(last_output)

        else:
            # Fallback: single op
            op = random.choice([OpType.RELU, OpType.GELU])
            ops.append((op, [last_output]))
            last_output = len(ops) - 1

        # Occasionally add a skip connection to earlier node
        if len(skip_sources) > 2 and random.random() < 0.3:
            skip_from = random.choice(skip_sources[:-1])
            ops.append((OpType.ADD, [skip_from, last_output]))
            last_output = len(ops) - 1

    return ops[:target_ops]


def _make_hardware(
    difficulty: str,
    shape: list[int],
    n_ops: int,
) -> HardwareSpec:
    """
    Create hardware spec that ensures the right difficulty.
    Key: fast_mem must be tight enough to force decisions but not impossible.
    """
    single_tensor_bytes = shape[0] * shape[1] * 4  # float32

    # CRITICAL TUNING: bandwidth must be low enough that reloading a tensor
    # is a significant fraction of total step cost. This is what creates
    # the retention learning signal.
    #
    # With shape=[256,256], tensor = 262144 bytes
    # A reload at bw=10 costs 26214 cycles
    # A pointwise op with flops=0.1*65536=6554 cycles
    # So reload >> compute for pointwise -> retention matters!
    #
    # A matmul with flops=4*65536=262144 cycles
    # Reload at bw=10 costs 26214 cycles -> ~10% overhead
    # Saving 2-3 reloads via retention saves 20-30%

    if difficulty == "easy":
        capacity = single_tensor_bytes * 6
        bw = 15.0           # moderate bandwidth
        launch_cost = 2000.0
        max_fusion = 8
    elif difficulty == "medium":
        capacity = single_tensor_bytes * 4
        bw = 10.0           # slow: reloads are expensive
        launch_cost = 3000.0
        max_fusion = 6
    elif difficulty == "hard":
        capacity = single_tensor_bytes * 3
        bw = 8.0            # very slow: every reload hurts
        launch_cost = 4000.0
        max_fusion = 5
    else:  # expert
        capacity = int(single_tensor_bytes * 2.5)
        bw = 6.0            # extremely slow: retention is critical
        launch_cost = 5000.0
        max_fusion = 4

    return HardwareSpec(
        fast_mem_capacity=capacity,
        slow_mem_bandwidth=bw,
        kernel_launch_cost=launch_cost,
        max_fusion_depth=max_fusion,
    )


# ============================================================
# Task Definitions (fixed seeds for evaluation)
# ============================================================

TASKS = {
    "task1_chain": {
        "num_ops": 8, "difficulty": "easy", "seed": 42,
        "topology": "linear", "shape": [128, 128],
        "description": "Linear chain of 8 ops. Learn basic fusion.",
        "max_steps": 8,
    },
    "task2_residual": {
        "num_ops": 12, "difficulty": "medium", "seed": 123,
        "topology": "residual", "shape": [256, 256],
        "description": "Residual blocks. Learn retention for skip connections.",
        "max_steps": 12,
    },
    "task3_attention": {
        "num_ops": 16, "difficulty": "hard", "seed": 456,
        "topology": "attention", "shape": [256, 256],
        "description": "Attention pattern. Learn fan-out retention + fusion boundaries.",
        "max_steps": 16,
    },
    "task4_mixed": {
        "num_ops": 24, "difficulty": "expert", "seed": 789,
        "topology": None, "shape": [256, 256],
        "description": "Mixed topology with skip connections. Full strategy required.",
        "max_steps": 24,
    },
    "task5_adversarial": {
        "num_ops": 20, "difficulty": "hard", "seed": 1337,
        "topology": "adversarial", "shape": [256, 256],
        "description": "Adversarial: long skips + multi-reuse. Only retention-aware policy wins.",
        "max_steps": 20,
    },
}


def load_task(task_name: str) -> tuple[Graph, dict]:
    """Load a fixed task. Returns (graph, config)."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    cfg = TASKS[task_name]
    graph = generate_graph(
        num_ops=cfg["num_ops"],
        shape=cfg.get("shape"),
        difficulty=cfg["difficulty"],
        seed=cfg["seed"],
        topology=cfg.get("topology"),
    )
    return graph, cfg


def generate_training_graph(curriculum_level: float = 0.5) -> Graph:
    """
    Generate a random graph for training.
    curriculum_level in [0, 1]: 0 = easy, 1 = hard.

    Distribution:
        0.0-0.3: 8-16 ops, easy/medium
        0.3-0.7: 16-32 ops, medium/hard
        0.7-1.0: 32-50 ops, hard/expert
    """
    if curriculum_level < 0.3:
        n_ops = random.randint(8, 16)
        difficulty = random.choice(["easy", "medium"])
    elif curriculum_level < 0.7:
        n_ops = random.randint(16, 32)
        difficulty = random.choice(["medium", "hard"])
    else:
        n_ops = random.randint(32, 50)
        difficulty = random.choice(["hard", "expert"])

    shape_dim = random.choice([128, 256, 512])
    return generate_graph(
        num_ops=n_ops,
        shape=[shape_dim, shape_dim],
        difficulty=difficulty,
    )


def list_tasks() -> list[str]:
    return list(TASKS.keys())
