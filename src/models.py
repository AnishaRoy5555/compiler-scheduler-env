"""
FusionOps V2 - Data Models
Compact, typed models designed for LLM-native observation and action spaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OpType(str, Enum):
    MATMUL = "matmul"
    CONV2D = "conv2d"
    RELU = "relu"
    GELU = "gelu"
    LAYERNORM = "layernorm"
    SOFTMAX = "softmax"
    ADD = "add"          # residual add
    TRANSPOSE = "transpose"
    REDUCE = "reduce"    # reduce-sum/mean


# Op categories for cost model
POINTWISE_OPS = {OpType.RELU, OpType.GELU, OpType.ADD, OpType.TRANSPOSE}
REDUCTION_OPS = {OpType.LAYERNORM, OpType.SOFTMAX, OpType.REDUCE}
COMPUTE_OPS = {OpType.MATMUL, OpType.CONV2D}


@dataclass
class Node:
    """A single operation in the computation graph."""
    id: int
    op: OpType
    shape: list[int]          # output shape, e.g. [256, 256]
    inputs: list[int]         # IDs of predecessor nodes (NOT tensor IDs)
    flops: float              # base compute cost in cycles
    output_bytes: int         # output tensor size in bytes
    input_bytes: list[int]    # per-input tensor sizes


@dataclass
class HardwareSpec:
    fast_mem_capacity: int    # bytes
    slow_mem_bandwidth: float # bytes/cycle
    kernel_launch_cost: float # fixed overhead per kernel launch
    max_fusion_depth: int     # max ops per fused kernel


@dataclass
class Graph:
    """Computation graph with hardware constraints."""
    nodes: list[Node]
    hardware: HardwareSpec
    # Derived
    _successors: dict[int, list[int]] = field(default_factory=dict, repr=False)
    _consumers_count: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_derived()

    def _build_derived(self):
        self._successors = {n.id: [] for n in self.nodes}
        self._consumers_count = {n.id: 0 for n in self.nodes}
        for n in self.nodes:
            for pred_id in n.inputs:
                self._successors[pred_id].append(n.id)
                self._consumers_count[pred_id] = self._consumers_count.get(pred_id, 0) + 1

    def successors(self, node_id: int) -> list[int]:
        return self._successors.get(node_id, [])

    def future_uses(self, node_id: int) -> int:
        """How many downstream ops still need this node's output."""
        return self._consumers_count.get(node_id, 0)

    def topo_order(self) -> list[int]:
        """Return nodes in topological order."""
        from collections import deque
        in_degree = {}
        for n in self.nodes:
            in_degree[n.id] = len(n.inputs)
        queue = deque(nid for nid, d in in_degree.items() if d == 0)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._successors[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order


@dataclass
class Action:
    """Agent's per-step decision."""
    fuse_with_prev: bool       # fuse current node with the previous fusion group
    tile: int                  # tile size from discrete set {32, 64, 128, 256}
    retain: list[int]          # node IDs whose outputs to keep in fast memory

    VALID_TILES = [32, 64, 128, 256]


@dataclass
class FusionGroup:
    """A group of ops that will execute as one kernel."""
    node_ids: list[int]
    tile: int
    retained: list[int]
    latency: float = 0.0


@dataclass
class ScheduleState:
    """Mutable state for an episode."""
    step: int = 0
    current_node_idx: int = 0       # index into topo_order
    fusion_groups: list[FusionGroup] = field(default_factory=list)
    fast_mem_contents: set[int] = field(default_factory=set)  # node IDs in fast mem
    fast_mem_used: int = 0
    total_latency: float = 0.0
    total_reloads: int = 0
    total_kernel_launches: int = 0
    # Track remaining uses for retention decisions
    remaining_uses: dict[int, int] = field(default_factory=dict)
    # Current fusion group being built
    current_group: Optional[FusionGroup] = None

    def clone(self) -> ScheduleState:
        import copy
        return copy.deepcopy(self)
