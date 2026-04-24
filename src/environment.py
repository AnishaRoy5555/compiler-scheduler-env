"""
FusionOps V2 - Environment

Sequential step environment:
- Walk graph in topo order
- At each node, agent decides: fuse_with_prev, tile, retain
- Episode ends when all nodes are scheduled
- Reward = improvement over greedy baseline + step signals

Observation: compact JSON (not prose)
Action: JSON with fuse_with_prev, tile, retain
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    Graph, Node, Action, FusionGroup, ScheduleState, OpType,
    POINTWISE_OPS, COMPUTE_OPS,
)
from .cost_model import (
    compute_group_cost, compute_greedy_baseline, compute_naive_baseline,
    CostBreakdown,
)


@dataclass
class StepResult:
    observation: str      # compact JSON string
    reward: float
    done: bool
    info: dict = field(default_factory=dict)
    score: Optional[float] = None


class FusionOpsEnv:
    """
    RL environment for ML computation graph scheduling.

    Sequential decisions: walk topo order, at each node decide
    fuse/tile/retain. One step per node.
    """

    def __init__(self, graph: Graph, max_steps: Optional[int] = None):
        self.graph = graph
        self.max_steps = max_steps or len(graph.nodes)
        self.topo_order = graph.topo_order()

        # Precompute baselines
        self.greedy_latency = compute_greedy_baseline(graph)
        self.naive_latency = compute_naive_baseline(graph)

        self.state: Optional[ScheduleState] = None

    def reset(self) -> StepResult:
        """Initialize episode. Returns first observation."""
        self.state = ScheduleState()
        self.state.remaining_uses = {
            n.id: self.graph.future_uses(n.id) for n in self.graph.nodes
        }
        self.state.current_node_idx = 0
        self.state.current_group = FusionGroup(
            node_ids=[], tile=128, retained=[]
        )

        obs = self._make_observation()
        return StepResult(observation=obs, reward=0.0, done=False)

    def step(self, action: Action) -> StepResult:
        """Execute one step: process the current node with the given action."""
        assert self.state is not None, "Must call reset() first"

        if self.state.current_node_idx >= len(self.topo_order):
            return StepResult(
                observation=self._make_observation(),
                reward=0.0,
                done=True,
                score=self.get_score(),
            )

        current_nid = self.topo_order[self.state.current_node_idx]
        current_node = self.graph.nodes[current_nid]
        hw = self.graph.hardware

        # Validate tile
        if action.tile not in Action.VALID_TILES:
            action.tile = 128  # default

        # Validate retain: can only retain nodes whose outputs exist
        valid_retain = []
        for r in action.retain:
            if 0 <= r < len(self.graph.nodes):
                valid_retain.append(r)
        action.retain = valid_retain

        step_reward = 0.0
        error = None

        if action.fuse_with_prev and self.state.current_group.node_ids:
            # Try to fuse with current group
            proposed_ids = self.state.current_group.node_ids + [current_nid]

            # Check connectivity
            connected = any(
                inp in self.state.current_group.node_ids
                for inp in current_node.inputs
            )

            # Check fusion depth limit
            within_limit = len(proposed_ids) <= hw.max_fusion_depth

            if connected and within_limit:
                # Test if the fused group is valid (memory check)
                test_group = FusionGroup(
                    node_ids=proposed_ids,
                    tile=action.tile,
                    retained=action.retain,
                )
                cost = compute_group_cost(self.graph, test_group, self.state)

                if cost.is_valid:
                    # Successful fusion
                    self.state.current_group = test_group
                    step_reward += 0.02  # small reward for valid fusion
                else:
                    # OOM: can't fuse, finalize current group and start new
                    error = cost.error
                    step_reward -= 0.01  # small penalty for OOM attempt
                    self._finalize_current_group()
                    self.state.current_group = FusionGroup(
                        node_ids=[current_nid],
                        tile=action.tile,
                        retained=action.retain,
                    )
            else:
                # Not connected or exceeds depth: finalize and start new
                if not connected:
                    error = "Cannot fuse: not connected to current group"
                else:
                    error = f"Cannot fuse: exceeds max fusion depth {hw.max_fusion_depth}"
                step_reward -= 0.005
                self._finalize_current_group()
                self.state.current_group = FusionGroup(
                    node_ids=[current_nid],
                    tile=action.tile,
                    retained=action.retain,
                )
        else:
            # Don't fuse: finalize current group (if any) and start new
            self._finalize_current_group()
            self.state.current_group = FusionGroup(
                node_ids=[current_nid],
                tile=action.tile,
                retained=action.retain,
            )

        # Update remaining uses for consumed inputs
        for inp_id in current_node.inputs:
            if inp_id in self.state.remaining_uses:
                self.state.remaining_uses[inp_id] -= 1
                # If this was the last use, the tensor can be freed
                if self.state.remaining_uses[inp_id] <= 0:
                    self.state.fast_mem_contents.discard(inp_id)

        # Retention step reward
        for r in action.retain:
            remaining = self.state.remaining_uses.get(r, 0)
            if remaining > 0:
                step_reward += 0.01  # good: retaining something that will be reused
            else:
                step_reward -= 0.01  # bad: retaining dead tensor, wastes memory

        # Reload penalty
        for inp_id in current_node.inputs:
            if inp_id not in self.state.fast_mem_contents and inp_id >= 0:
                # Had to reload from slow memory
                self.state.total_reloads += 1
                step_reward -= 0.005

        # Advance
        self.state.step += 1
        self.state.current_node_idx += 1

        # Check if episode is done
        done = self.state.current_node_idx >= len(self.topo_order)
        if done:
            # Finalize last group
            self._finalize_current_group()

        # Check step limit
        if self.state.step >= self.max_steps and not done:
            done = True
            self._finalize_current_group()

        score = self.get_score() if done else None

        # Final reward
        if done:
            # Episode completion reward based on improvement over greedy
            improvement = (self.greedy_latency - self.state.total_latency) / self.greedy_latency
            step_reward += max(0, improvement) * 0.5  # big bonus for beating greedy
            if improvement < 0:
                step_reward += improvement * 0.1  # smaller penalty for being worse

        obs = self._make_observation(error=error)
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            info={
                "total_latency": self.state.total_latency,
                "greedy_latency": self.greedy_latency,
                "kernel_launches": self.state.total_kernel_launches,
                "reloads": self.state.total_reloads,
                "error": error,
            },
            score=score,
        )

    def _finalize_current_group(self):
        """Finalize the current fusion group: compute cost, update state."""
        if not self.state.current_group or not self.state.current_group.node_ids:
            return

        group = self.state.current_group
        cost = compute_group_cost(self.graph, group, self.state)

        if cost.is_valid:
            group.latency = cost.total_latency
            self.state.total_latency += cost.total_latency
            self.state.total_kernel_launches += 1

            # Update fast memory
            # Remove tensors not retained
            new_fast_mem = set()
            for r in group.retained:
                new_fast_mem.add(r)
            # Keep previously retained tensors that still have uses
            for existing in self.state.fast_mem_contents:
                if self.state.remaining_uses.get(existing, 0) > 0:
                    new_fast_mem.add(existing)
            self.state.fast_mem_contents = new_fast_mem

            # Compute fast mem used
            self.state.fast_mem_used = sum(
                self.graph.nodes[nid].output_bytes
                for nid in self.state.fast_mem_contents
            )
        else:
            # Fallback: execute each op individually
            for nid in group.node_ids:
                solo = FusionGroup(node_ids=[nid], tile=128, retained=[])
                sc = compute_group_cost(self.graph, solo, self.state)
                if sc.is_valid:
                    self.state.total_latency += sc.total_latency
                else:
                    self.state.total_latency += 1e6  # catastrophic failure
                self.state.total_kernel_launches += 1

        self.state.fusion_groups.append(group)
        self.state.current_group = FusionGroup(
            node_ids=[], tile=128, retained=[]
        )

    def _make_observation(self, error: Optional[str] = None) -> str:
        """
        Compact JSON observation.
        Designed for LLM consumption: short, structured, stable schema.

        Step 0: full graph (compact per-node: id, op, inputs only -- no shape
                 unless graph has >20 nodes, then only show first 20 + summary)
        Steps 1+: current node + 2-node lookahead only
        """
        state = self.state

        # Current node info
        if state.current_node_idx < len(self.topo_order):
            current_nid = self.topo_order[state.current_node_idx]
            current_node = self.graph.nodes[current_nid]
        else:
            current_nid = -1
            current_node = None

        # Future uses for visible tensors (only non-zero)
        future_uses = {}
        for nid, uses in state.remaining_uses.items():
            if uses > 0:
                future_uses[str(nid)] = uses

        # Current fusion group
        current_group_info = {
            "node_ids": state.current_group.node_ids if state.current_group else [],
            "tile": state.current_group.tile if state.current_group else 128,
        }

        obs = {
            "step": state.step,
            "current_node": current_nid,
            "current_group": current_group_info,
            "fast_mem": sorted(state.fast_mem_contents),
            "fast_mem_used": state.fast_mem_used,
            "capacity": self.graph.hardware.fast_mem_capacity,
            "max_fusion": self.graph.hardware.max_fusion_depth,
            "future_uses": future_uses,
            "total_latency": round(state.total_latency, 1),
            "kernel_launches": state.total_kernel_launches,
            "greedy_baseline": round(self.greedy_latency, 1),
        }

        if error:
            obs["error"] = error

        # Step 0: include graph structure (compact)
        if state.step == 0:
            n_nodes = len(self.graph.nodes)
            # For large graphs (>20 nodes), only include shape for the first node
            # and omit shape for the rest to save tokens
            nodes_data = []
            for node in self.graph.nodes:
                entry = {
                    "id": node.id,
                    "op": node.op.value,
                    "inputs": node.inputs,
                }
                if n_nodes <= 20 or node.id == 0:
                    entry["shape"] = node.shape
                nodes_data.append(entry)
            obs["nodes"] = nodes_data
            obs["tensor_bytes"] = self.graph.nodes[0].output_bytes
        else:
            # Steps 1+: only current node + lookahead
            if current_node:
                obs["current_node_info"] = {
                    "id": current_nid,
                    "op": current_node.op.value,
                    "shape": current_node.shape,
                    "inputs": current_node.inputs,
                    "output_bytes": current_node.output_bytes,
                }
                # Show next 2 nodes for lookahead
                lookahead = []
                for offset in range(1, 3):
                    next_idx = state.current_node_idx + offset
                    if next_idx < len(self.topo_order):
                        next_nid = self.topo_order[next_idx]
                        nn = self.graph.nodes[next_nid]
                        lookahead.append({
                            "id": next_nid,
                            "op": nn.op.value,
                            "inputs": nn.inputs,
                        })
                if lookahead:
                    obs["lookahead"] = lookahead

        return json.dumps(obs, separators=(",", ":"))

    def get_score(self) -> float:
        """
        Score in [0, 1].
        0.0 = same as greedy baseline
        1.0 = 20%+ improvement over greedy (calibrated to achievable ceiling)
        Negative if worse than greedy.
        """
        if self.state is None or self.state.total_latency <= 0:
            return 0.0

        improvement = (self.greedy_latency - self.state.total_latency) / self.greedy_latency

        # Map to [0, 1]: 0% -> 0.0, 20%+ -> 1.0
        # Retention gap tests show 12-18% is achievable, so 20% = perfect
        score = improvement / 0.2
        return max(-1.0, min(1.0, score))

    def get_state(self) -> dict:
        """Full state for debugging."""
        assert self.state is not None
        return {
            "step": self.state.step,
            "current_node_idx": self.state.current_node_idx,
            "total_latency": self.state.total_latency,
            "greedy_latency": self.greedy_latency,
            "naive_latency": self.naive_latency,
            "kernel_launches": self.state.total_kernel_launches,
            "reloads": self.state.total_reloads,
            "fast_mem_contents": sorted(self.state.fast_mem_contents),
            "fast_mem_used": self.state.fast_mem_used,
            "fusion_groups": [
                {
                    "node_ids": g.node_ids,
                    "tile": g.tile,
                    "retained": g.retained,
                    "latency": g.latency,
                }
                for g in self.state.fusion_groups
            ],
            "score": self.get_score(),
        }


def parse_action(text: str) -> Optional[Action]:
    """
    Parse action from LLM output.
    Accepts JSON: {"fuse_with_prev": true, "tile": 128, "retain": [1, 3]}
    Also accepts natural language fallback.
    """
    text = text.strip()

    # Try JSON parse first
    try:
        # Handle markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.startswith("```"))
            text = text.strip()

        data = json.loads(text)
        return Action(
            fuse_with_prev=bool(data.get("fuse_with_prev", False)),
            tile=int(data.get("tile", 128)),
            retain=list(data.get("retain", [])),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: parse key=value format
    import re
    fuse = False
    tile = 128
    retain = []

    fuse_match = re.search(r'fuse[_\s]*(?:with[_\s]*prev)?[=:\s]*(true|false|yes|no|1|0)', text, re.I)
    if fuse_match:
        fuse = fuse_match.group(1).lower() in ("true", "yes", "1")

    tile_match = re.search(r'tile[=:\s]*(\d+)', text, re.I)
    if tile_match:
        tile = int(tile_match.group(1))

    retain_match = re.search(r'retain[=:\s]*\[([^\]]*)\]', text, re.I)
    if retain_match:
        retain_str = retain_match.group(1).strip()
        if retain_str:
            retain = [int(x.strip()) for x in retain_str.split(",") if x.strip()]

    return Action(fuse_with_prev=fuse, tile=tile, retain=retain)
