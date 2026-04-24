"""
FusionOps V2 - Physics-Based Cost Model

Computes latency for a fusion group using:
1. Kernel launch overhead (fixed per group)
2. Compute cost (sum of flops / throughput)
3. Memory transfer cost (loads from slow mem / bandwidth)
4. Fusion benefit: intermediate tensors between fused ops are ephemeral (zero transfer)
5. Retention benefit: retained tensors avoid reload in future steps
6. Tiling: smaller tiles reduce peak working set but may increase total transfers

Roofline: latency = max(compute_time, memory_time) + kernel_launch_cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import (
    Graph, Node, FusionGroup, ScheduleState, Action, OpType,
    POINTWISE_OPS, REDUCTION_OPS, COMPUTE_OPS,
)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a fusion group."""
    compute_cycles: float
    memory_load_cycles: float
    memory_store_cycles: float
    kernel_launch_cycles: float
    total_latency: float
    peak_working_set: int     # bytes
    tensors_loaded: list[int]  # node IDs loaded from slow mem
    tensors_stored: list[int]  # node IDs evicted to slow mem
    is_valid: bool
    error: Optional[str] = None


def compute_group_cost(
    graph: Graph,
    group: FusionGroup,
    state: ScheduleState,
) -> CostBreakdown:
    """
    Compute the full cost of executing a fusion group.

    Physics:
    - Each node in the group executes sequentially within the kernel
    - Inputs to the group that are NOT produced by another node in the group
      must be loaded from memory (unless already in fast_mem -> free)
    - Outputs of the group that are NOT consumed by another node in the group
      must be stored (unless retained in fast_mem)
    - Intermediate tensors (produced and consumed within the group) are ephemeral
    """
    hw = graph.hardware
    node_ids = set(group.node_ids)
    nodes = [graph.nodes[nid] for nid in group.node_ids]

    # --- Classify tensors ---
    # Boundary inputs: nodes consumed by this group but not produced in it
    boundary_inputs: list[int] = []
    # Ephemeral: produced AND consumed within the group
    ephemeral: set[int] = set()
    # Boundary outputs: produced by group, NOT consumed within group
    boundary_outputs: list[int] = []

    produced_in_group = set(group.node_ids)
    consumed_in_group: set[int] = set()

    for node in nodes:
        for inp_id in node.inputs:
            consumed_in_group.add(inp_id)
            if inp_id not in produced_in_group:
                boundary_inputs.append(inp_id)

    for nid in group.node_ids:
        successors_in_group = [s for s in graph.successors(nid) if s in node_ids]
        if successors_in_group and nid in consumed_in_group:
            # This node's output is consumed within the group
            ephemeral.add(nid)
        else:
            # Output leaves the group
            if any(s in node_ids for s in graph.successors(nid)):
                ephemeral.add(nid)
            else:
                boundary_outputs.append(nid)

    # Also check: nodes whose output is consumed BOTH inside and outside
    for nid in group.node_ids:
        all_succs = graph.successors(nid)
        succs_in = [s for s in all_succs if s in node_ids]
        succs_out = [s for s in all_succs if s not in node_ids]
        if succs_in and succs_out:
            # Consumed both inside and outside: it's ephemeral inside
            # but still needs to be available outside
            if nid not in boundary_outputs:
                boundary_outputs.append(nid)
        elif not succs_in and nid in produced_in_group:
            if nid not in boundary_outputs:
                boundary_outputs.append(nid)

    # Deduplicate
    boundary_inputs = list(dict.fromkeys(boundary_inputs))
    boundary_outputs = list(dict.fromkeys(boundary_outputs))

    # --- Compute cost ---
    tile = group.tile
    total_flops = 0.0
    for node in nodes:
        elements = node.shape[0] * node.shape[1]
        # Tiling overhead: number of tiles
        n_tiles = max(1, (elements + tile * tile - 1) // (tile * tile))

        if node.op in COMPUTE_OPS:
            # COMPUTE OPS: tile size significantly affects efficiency
            # Larger tiles = better arithmetic intensity (less overhead per element)
            # Smaller tiles = more tile boundary overhead + worse cache behavior
            # This creates a real tradeoff: large tile for compute efficiency
            # vs small tile to fit in memory when fusing
            efficiency = min(1.0, tile / 128.0)  # 128 is "native" efficiency
            overhead = 1.0 / (efficiency ** 0.5)  # sqrt penalty for small tiles
            # Additional per-tile startup cost for compute ops
            tile_startup = n_tiles * (node.flops * 0.02)  # 2% startup per tile
            total_flops += node.flops * overhead + tile_startup
        elif node.op in REDUCTION_OPS:
            # Reduction ops: moderate tile sensitivity
            overhead = 1.0 + 0.1 * max(0, 3 - n_tiles)  # small tiles less efficient
            total_flops += node.flops * overhead
        else:
            # Pointwise: tile-insensitive (element-independent)
            total_flops += node.flops

    # Compute throughput: 1 flop/cycle (normalized)
    compute_cycles = total_flops

    # --- Memory cost ---
    bw = hw.slow_mem_bandwidth
    tensors_loaded = []
    load_bytes = 0

    for inp_id in boundary_inputs:
        if inp_id in state.fast_mem_contents:
            # Already in fast mem, zero cost
            continue
        # Must load from slow memory
        inp_node = graph.nodes[inp_id]
        load_bytes += inp_node.output_bytes
        tensors_loaded.append(inp_id)

    # Also load graph inputs (nodes with no predecessors consumed by this group)
    for node in nodes:
        for inp_id in node.inputs:
            if inp_id < 0:  # graph input sentinel
                continue

    memory_load_cycles = load_bytes / bw if bw > 0 else 0.0

    # Store outputs not retained
    tensors_stored = []
    store_bytes = 0
    retained_set = set(group.retained)

    for out_id in boundary_outputs:
        if out_id in retained_set:
            # Kept in fast mem, no store cost
            continue
        out_node = graph.nodes[out_id]
        store_bytes += out_node.output_bytes
        tensors_stored.append(out_id)

    memory_store_cycles = store_bytes / bw if bw > 0 else 0.0

    # --- Working set check ---
    peak_ws = 0
    # Inputs in fast mem (loaded or already there)
    for inp_id in boundary_inputs:
        inp_node = graph.nodes[inp_id]
        peak_ws += min(inp_node.output_bytes, tile * tile * 4)  # tiled slice

    # Outputs being produced (one at a time, but accumulator needed)
    for out_id in boundary_outputs:
        out_node = graph.nodes[out_id]
        peak_ws += min(out_node.output_bytes, tile * tile * 4)

    # Already-retained tensors occupying fast mem
    for ret_id in state.fast_mem_contents:
        if ret_id not in node_ids:  # not being consumed by this group
            ret_node = graph.nodes[ret_id]
            peak_ws += ret_node.output_bytes

    # New retentions
    for ret_id in group.retained:
        if ret_id not in state.fast_mem_contents:
            ret_node = graph.nodes[ret_id]
            peak_ws += ret_node.output_bytes

    if peak_ws > hw.fast_mem_capacity:
        return CostBreakdown(
            compute_cycles=0,
            memory_load_cycles=0,
            memory_store_cycles=0,
            kernel_launch_cycles=0,
            total_latency=0,
            peak_working_set=peak_ws,
            tensors_loaded=[],
            tensors_stored=[],
            is_valid=False,
            error=f"OOM: working set {peak_ws} > capacity {hw.fast_mem_capacity}",
        )

    # --- Total latency (roofline) ---
    mem_cycles = memory_load_cycles + memory_store_cycles
    roofline = max(compute_cycles, mem_cycles)
    total = roofline + hw.kernel_launch_cost

    return CostBreakdown(
        compute_cycles=compute_cycles,
        memory_load_cycles=memory_load_cycles,
        memory_store_cycles=memory_store_cycles,
        kernel_launch_cycles=hw.kernel_launch_cost,
        total_latency=total,
        peak_working_set=peak_ws,
        tensors_loaded=tensors_loaded,
        tensors_stored=tensors_stored,
        is_valid=True,
    )


def compute_greedy_baseline(graph: Graph) -> float:
    """
    DELIBERATELY MYOPIC greedy baseline:
    - Walk topo order
    - Fuse with prev if connected and fits in memory
    - Retain ONLY if the very next node in topo order directly consumes this output
    - Ignores all longer-range skip connections (this is the weakness RL exploits)
    - Fixed tile = 128

    This baseline is beatable by 10-25% on graphs with skip connections,
    creating clear reward signal for retention-aware policies.
    """
    topo = graph.topo_order()
    hw = graph.hardware
    state = ScheduleState()
    state.remaining_uses = {n.id: graph.future_uses(n.id) for n in graph.nodes}

    total_latency = 0.0
    current_group_ids: list[int] = []
    current_group_size = 0  # estimated working set

    for idx, nid in enumerate(topo):
        node = graph.nodes[nid]

        # Try to fuse with current group
        can_fuse = False
        if current_group_ids:
            # Check: is this node connected to the current group?
            connected = any(inp in current_group_ids for inp in node.inputs)
            if connected and len(current_group_ids) < hw.max_fusion_depth:
                # Estimate working set if we add this node
                est_ws = current_group_size + node.output_bytes
                if est_ws <= hw.fast_mem_capacity * 0.8:  # 80% threshold
                    can_fuse = True

        if can_fuse:
            current_group_ids.append(nid)
            current_group_size += node.output_bytes
        else:
            # Finalize current group
            if current_group_ids:
                # MYOPIC RETENTION: only retain if the VERY NEXT topo node consumes it
                retained = []
                next_nid = topo[idx] if idx < len(topo) else -1
                if next_nid >= 0:
                    next_node = graph.nodes[next_nid]
                    for gid in current_group_ids:
                        if gid in next_node.inputs:
                            retained.append(gid)

                group = FusionGroup(
                    node_ids=current_group_ids,
                    tile=128,
                    retained=retained,
                )
                cost = compute_group_cost(graph, group, state)
                if cost.is_valid:
                    total_latency += cost.total_latency
                    # Update state
                    state.fast_mem_contents = set(retained)
                else:
                    # Fallback: execute each op individually
                    for gid in current_group_ids:
                        solo = FusionGroup(node_ids=[gid], tile=128, retained=[])
                        sc = compute_group_cost(graph, solo, state)
                        total_latency += sc.total_latency if sc.is_valid else 1e6
                        state.fast_mem_contents.clear()

            # Start new group
            current_group_ids = [nid]
            current_group_size = node.output_bytes

    # Finalize last group
    if current_group_ids:
        group = FusionGroup(
            node_ids=current_group_ids,
            tile=128,
            retained=[],
        )
        cost = compute_group_cost(graph, group, state)
        if cost.is_valid:
            total_latency += cost.total_latency
        else:
            for gid in current_group_ids:
                solo = FusionGroup(node_ids=[gid], tile=128, retained=[])
                sc = compute_group_cost(graph, solo, state)
                total_latency += sc.total_latency if sc.is_valid else 1e6

    return total_latency


def compute_naive_baseline(graph: Graph) -> float:
    """
    Worst-case baseline: every op individually, no fusion, no retention.
    """
    state = ScheduleState()
    total = 0.0
    for node in graph.nodes:
        group = FusionGroup(node_ids=[node.id], tile=128, retained=[])
        cost = compute_group_cost(graph, group, state)
        total += cost.total_latency if cost.is_valid else 1e6
    return total
