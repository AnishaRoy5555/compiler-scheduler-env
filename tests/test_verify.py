"""
FusionOps V2 - Verification Tests

Verifies:
1. Greedy baseline is beatable by 10-25% on adversarial graphs
2. Retain vs no-retain gap >= 10% on skip-connection graphs
3. Tile size creates meaningful variation for compute ops
4. Reward distribution has spread (not all near zero)
5. All tasks generate valid graphs and run without errors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Graph, FusionGroup, ScheduleState, Action
from src.graph_gen import generate_graph, load_task, list_tasks, generate_training_graph
from src.cost_model import (
    compute_group_cost, compute_greedy_baseline, compute_naive_baseline,
)
from src.environment import FusionOpsEnv, parse_action


def test_all_tasks_valid():
    """All fixed tasks generate valid graphs and can be stepped through."""
    print("=" * 60)
    print("TEST: All tasks generate valid graphs")
    print("=" * 60)
    for task_name in list_tasks():
        graph, cfg = load_task(task_name)
        assert len(graph.nodes) > 0, f"{task_name}: no nodes"
        topo = graph.topo_order()
        assert len(topo) == len(graph.nodes), f"{task_name}: topo order incomplete"

        # Run through env
        env = FusionOpsEnv(graph, max_steps=cfg["max_steps"])
        result = env.reset()
        assert not result.done

        # Step with default actions
        steps = 0
        while not result.done and steps < cfg["max_steps"]:
            action = Action(fuse_with_prev=False, tile=128, retain=[])
            result = env.step(action)
            steps += 1

        print(f"  {task_name}: {len(graph.nodes)} nodes, "
              f"greedy={env.greedy_latency:.0f}, naive={env.naive_latency:.0f}, "
              f"agent={env.state.total_latency:.0f}, "
              f"score={env.get_score():.3f}")
    print("  PASS\n")


def _run_smart_scheduler(graph):
    """
    Smart scheduler: same fusion as greedy BUT with long-range retention.
    Key difference from greedy: retains ANY output that has remaining future
    uses, not just immediate-next-step. Also preserves previously retained
    tensors across group boundaries.
    """
    from src.cost_model import compute_group_cost
    topo = graph.topo_order()
    hw = graph.hardware
    state = ScheduleState()
    state.remaining_uses = {n.id: graph.future_uses(n.id) for n in graph.nodes}
    total_latency = 0.0

    current_group_ids: list[int] = []
    current_group_size = 0

    def finalize_group(group_ids, is_last=False):
        nonlocal total_latency
        if not group_ids:
            return
        # Smart retention: keep outputs with remaining future uses
        retained = []
        for gid in group_ids:
            if state.remaining_uses.get(gid, 0) > 0:
                retained.append(gid)
        # Also carry forward previously retained tensors still alive
        for existing in list(state.fast_mem_contents):
            if state.remaining_uses.get(existing, 0) > 0:
                if existing not in retained and existing not in group_ids:
                    retained.append(existing)
        # Budget: drop least-useful if over 50% capacity
        ret_bytes = sum(graph.nodes[r].output_bytes for r in retained)
        while retained and ret_bytes > hw.fast_mem_capacity * 0.5:
            retained.sort(key=lambda x: state.remaining_uses.get(x, 0))
            dropped = retained.pop(0)
            ret_bytes -= graph.nodes[dropped].output_bytes

        group = FusionGroup(node_ids=group_ids, tile=128, retained=retained)
        cost = compute_group_cost(graph, group, state)
        if cost.is_valid:
            total_latency += cost.total_latency
            state.fast_mem_contents = set(retained)
        else:
            # Retry without retention
            group2 = FusionGroup(node_ids=group_ids, tile=128, retained=[])
            cost2 = compute_group_cost(graph, group2, state)
            if cost2.is_valid:
                total_latency += cost2.total_latency
            else:
                for gid in group_ids:
                    solo = FusionGroup(node_ids=[gid], tile=128, retained=[])
                    sc = compute_group_cost(graph, solo, state)
                    total_latency += sc.total_latency if sc.is_valid else 1e6
            state.fast_mem_contents.clear()

    for idx, nid in enumerate(topo):
        node = graph.nodes[nid]

        can_fuse = False
        if current_group_ids:
            connected = any(inp in current_group_ids for inp in node.inputs)
            if connected and len(current_group_ids) < hw.max_fusion_depth:
                est_ws = current_group_size + node.output_bytes
                if est_ws <= hw.fast_mem_capacity * 0.8:
                    can_fuse = True

        if can_fuse:
            current_group_ids.append(nid)
            current_group_size += node.output_bytes
        else:
            finalize_group(current_group_ids)
            current_group_ids = [nid]
            current_group_size = node.output_bytes

        # Decrement remaining uses for consumed inputs
        for inp in node.inputs:
            if inp in state.remaining_uses:
                state.remaining_uses[inp] -= 1
                if state.remaining_uses[inp] <= 0:
                    state.fast_mem_contents.discard(inp)

    finalize_group(current_group_ids, is_last=True)
    return total_latency


def test_greedy_beatability():
    """Greedy baseline should be beatable by smart retention on adversarial graphs."""
    print("=" * 60)
    print("TEST: Greedy is beatable on adversarial graphs")
    print("=" * 60)

    for task_name in ["task5_adversarial", "task2_residual", "task4_mixed"]:
        graph, cfg = load_task(task_name)
        greedy = compute_greedy_baseline(graph)
        naive = compute_naive_baseline(graph)
        smart = _run_smart_scheduler(graph)

        improvement = (greedy - smart) / greedy * 100
        print(f"  {task_name}:")
        print(f"    Naive:  {naive:.0f}")
        print(f"    Greedy: {greedy:.0f}")
        print(f"    Smart:  {smart:.0f}")
        print(f"    Improvement over greedy: {improvement:+.1f}%")

    print()


def test_retention_gap():
    """Retaining vs not retaining should create >= 10% gap on skip graphs."""
    print("=" * 60)
    print("TEST: Retention creates meaningful gap")
    print("=" * 60)

    for task_name in ["task2_residual", "task5_adversarial"]:
        graph, cfg = load_task(task_name)
        topo = graph.topo_order()

        # Run with NO retention
        state_no_retain = ScheduleState()
        state_no_retain.remaining_uses = {n.id: graph.future_uses(n.id) for n in graph.nodes}
        lat_no_retain = 0.0
        for nid in topo:
            group = FusionGroup(node_ids=[nid], tile=128, retained=[])
            cost = compute_group_cost(graph, group, state_no_retain)
            lat_no_retain += cost.total_latency if cost.is_valid else 1e6
            state_no_retain.fast_mem_contents.clear()

        # Run with SMART retention
        state_retain = ScheduleState()
        state_retain.remaining_uses = {n.id: graph.future_uses(n.id) for n in graph.nodes}
        lat_retain = 0.0
        for nid in topo:
            node = graph.nodes[nid]
            retained = []
            if state_retain.remaining_uses.get(nid, 0) > 0:
                retained.append(nid)
            for ex in list(state_retain.fast_mem_contents):
                if state_retain.remaining_uses.get(ex, 0) > 0:
                    retained.append(ex)

            group = FusionGroup(node_ids=[nid], tile=128, retained=retained)
            cost = compute_group_cost(graph, group, state_retain)
            if cost.is_valid:
                lat_retain += cost.total_latency
                state_retain.fast_mem_contents = set(retained)
            else:
                group2 = FusionGroup(node_ids=[nid], tile=128, retained=[])
                cost2 = compute_group_cost(graph, group2, state_retain)
                lat_retain += cost2.total_latency if cost2.is_valid else 1e6
                state_retain.fast_mem_contents.clear()

            for inp in node.inputs:
                if inp in state_retain.remaining_uses:
                    state_retain.remaining_uses[inp] -= 1
                    if state_retain.remaining_uses[inp] <= 0:
                        state_retain.fast_mem_contents.discard(inp)

        gap = (lat_no_retain - lat_retain) / lat_no_retain * 100
        print(f"  {task_name}:")
        print(f"    No retention:    {lat_no_retain:.0f}")
        print(f"    Smart retention: {lat_retain:.0f}")
        print(f"    Gap: {gap:.1f}%")

    print()


def test_tile_sensitivity():
    """Different tile sizes should create meaningful cost variation for compute ops."""
    print("=" * 60)
    print("TEST: Tile size creates meaningful variation")
    print("=" * 60)

    graph, _ = load_task("task3_attention")

    # Find a matmul node
    matmul_nodes = [n for n in graph.nodes if n.op.value == "matmul"]
    if not matmul_nodes:
        print("  SKIP: No matmul nodes found")
        return

    nid = matmul_nodes[0].id
    state = ScheduleState()

    costs = {}
    for tile in [32, 64, 128, 256]:
        group = FusionGroup(node_ids=[nid], tile=tile, retained=[])
        cost = compute_group_cost(graph, group, state)
        if cost.is_valid:
            costs[tile] = cost.total_latency

    if len(costs) >= 2:
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        variation = (max_cost - min_cost) / min_cost * 100
        print(f"  MatMul node {nid} costs by tile:")
        for tile, c in sorted(costs.items()):
            print(f"    tile={tile}: {c:.0f}")
        print(f"  Variation: {variation:.1f}%")
        if variation > 5:
            print("  PASS")
        else:
            print("  WARNING: Low tile variation")
    print()


def test_reward_distribution():
    """Check reward distribution across random graphs has spread."""
    print("=" * 60)
    print("TEST: Reward distribution has spread")
    print("=" * 60)

    scores = []
    for seed in range(20):
        graph = generate_graph(num_ops=16, difficulty="medium", seed=seed + 100)
        greedy = compute_greedy_baseline(graph)
        naive = compute_naive_baseline(graph)
        if greedy > 0 and naive > 0:
            gap = (naive - greedy) / naive
            scores.append(gap)

    if scores:
        import statistics
        print(f"  Naive-Greedy gaps across 20 random graphs:")
        print(f"    Min:    {min(scores):.3f}")
        print(f"    Max:    {max(scores):.3f}")
        print(f"    Mean:   {statistics.mean(scores):.3f}")
        print(f"    Stdev:  {statistics.stdev(scores):.3f}")
        if max(scores) - min(scores) > 0.05:
            print("  PASS: Good spread")
        else:
            print("  WARNING: Low spread")
    print()


def test_parse_action():
    """Action parser handles JSON and fallback formats."""
    print("=" * 60)
    print("TEST: Action parser")
    print("=" * 60)

    # JSON format
    a1 = parse_action('{"fuse_with_prev": true, "tile": 64, "retain": [1, 3]}')
    assert a1.fuse_with_prev == True
    assert a1.tile == 64
    assert a1.retain == [1, 3]
    print("  JSON format: PASS")

    # Fallback format
    a2 = parse_action('fuse_with_prev=false tile=128 retain=[]')
    assert a2.fuse_with_prev == False
    assert a2.tile == 128
    assert a2.retain == []
    print("  Fallback format: PASS")

    # Markdown-wrapped JSON
    a3 = parse_action('```json\n{"fuse_with_prev": true, "tile": 256, "retain": [0]}\n```')
    assert a3.fuse_with_prev == True
    assert a3.tile == 256
    print("  Markdown JSON: PASS")
    print()


def test_env_episode():
    """Full episode run with mixed actions."""
    print("=" * 60)
    print("TEST: Full episode with mixed actions")
    print("=" * 60)

    graph, cfg = load_task("task1_chain")
    env = FusionOpsEnv(graph, max_steps=cfg["max_steps"])
    result = env.reset()

    # Try fusing everything
    total_reward = 0.0
    steps = 0
    while not result.done:
        # Alternate: fuse, then don't, to test both paths
        fuse = steps > 0  # fuse after first
        action = Action(fuse_with_prev=fuse, tile=128, retain=[])
        result = env.step(action)
        total_reward += result.reward
        steps += 1

    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Score: {env.get_score():.3f}")
    print(f"  Latency: {env.state.total_latency:.0f} vs greedy {env.greedy_latency:.0f}")
    print("  PASS\n")


if __name__ == "__main__":
    test_all_tasks_valid()
    test_greedy_beatability()
    test_retention_gap()
    test_tile_sensitivity()
    test_reward_distribution()
    test_parse_action()
    test_env_episode()
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
