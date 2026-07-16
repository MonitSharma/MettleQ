"""MPO compression pipeline for CPU P9 peaked-circuit solving.

The algorithm follows the midpoint MPO cancellation and greedy unswapping
strategy introduced by Kremer and Dupuis, with CPU-focused engineering changes:
cached reusable SWAP MPOs, in-place circuit layer merging, and lower-cost
post-unswap rerouting.
"""

from quimb.tensor import MatrixProductOperator, Circuit, CircuitMPS

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit
from concurrent.futures import ThreadPoolExecutor

from .mpo import apply_circuit, apply_swaps, mpo_from_circuit

from .qiskit_utils import iter_layers, merge_layers, elem_counts, merge_gates, get_tn_info

import numpy as np
import time

import logging

IGNORED_WORK_OPS = {"swap", "measure", "barrier", "delay"}

# ------------------------------------------------------------------
#  Rewiring
# ------------------------------------------------------------------
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from qiskit.transpiler import CouplingMap

def rewire_layers(ls, perm, seed=None, sabre_trials=10000, sabre_heuristic="decay"):
    nq = len(perm)
    qc = merge_layers(ls)
    qc = QuantumCircuit(nq).compose(qc, qubits=np.argsort(perm))

    qc = ElidePermutations()(qc)
    ss = SabreSwap(
        coupling_map=CouplingMap.from_line(ls[0].num_qubits),
        heuristic=sabre_heuristic,
        trials=sabre_trials,
        seed=seed,
    )
    qc = ss(qc)

    return list(iter_layers(qc))


def _candidate_seeds(seed, count, stride):
    base = 0 if seed is None else int(seed)
    return [base + stride * idx for idx in range(count)]


def _route_static_stats(layers):
    body = layers[:-2] if len(layers) >= 2 else layers
    swap_counts = []
    work_counts = []
    for layer in body:
        ops = dict(layer.count_ops())
        swap_counts.append(ops.get("swap", 0))
        work_counts.append(count_work_ops_from_ops(ops))
    first_work_layer = next(
        (idx for idx, work_count in enumerate(work_counts) if work_count > 0),
        None,
    )
    work_layers = int(sum(1 for u in work_counts if u))
    return {
        "num_layers": len(body),
        "total_swaps": int(sum(swap_counts)),
        "max_layer_swaps": int(max(swap_counts, default=0)),
        "swap_only_layers": int(sum(1 for s, u in zip(swap_counts, work_counts) if s and not u)),
        "work_layers": work_layers,
        "unitary_layers": work_layers,
        "first_work_layer": first_work_layer,
        "leading_no_work_layers": int(
            len(body) if first_work_layer is None else first_work_layer
        ),
    }


def _route_bond_profile_score(layers, mpo_core, lookahead_layers, include_swaps=False):
    bonds = get_bond_sizes(mpo_core)
    if len(bonds) == 0:
        return (0, 0, 0, 0), {
            "bond_profile_layers_scored": 0,
            "bond_profile_total_cost": 0,
            "bond_profile_max_cost": 0,
            "bond_profile_gate_count": 0,
        }

    total_cost = 0
    max_cost = 0
    gate_count = 0
    layers_scored = 0
    body = layers[:-2] if len(layers) >= 2 else layers
    for layer in body:
        layer_cost = 0
        for q0, q1 in layer_two_qubit_pairs(layer, include_swaps=include_swaps):
            lo = min(q0, q1)
            hi = max(q0, q1)
            if hi <= lo:
                continue
            interval = bonds[lo:hi]
            if len(interval) == 0:
                continue
            cost = int(np.max(interval))
            layer_cost += cost
            max_cost = max(max_cost, cost)
            gate_count += 1
        if layer_cost:
            total_cost += layer_cost
            layers_scored += 1
            if layers_scored >= lookahead_layers:
                break

    return (
        total_cost,
        max_cost,
        gate_count,
        layers_scored,
    ), {
        "bond_profile_layers_scored": layers_scored,
        "bond_profile_total_cost": total_cost,
        "bond_profile_max_cost": max_cost,
        "bond_profile_gate_count": gate_count,
    }


def _score_routed_layers(
    layers,
    mpo_core,
    side,
    q2c,
    max_bond,
    cutoff,
    lookahead_layers,
    route_score,
):
    stats = _route_static_stats(layers)
    if route_score == "static":
        return (
            stats["leading_no_work_layers"],
            stats["swap_only_layers"],
            stats["max_layer_swaps"],
            stats["total_swaps"],
            stats["num_layers"],
        ), stats

    if route_score in ("bond_profile", "bond_profile_swaps"):
        score, bond_stats = _route_bond_profile_score(
            layers,
            mpo_core=mpo_core,
            lookahead_layers=lookahead_layers,
            include_swaps=route_score == "bond_profile_swaps",
        )
        stats.update(bond_stats)
        return score + (
            stats["swap_only_layers"],
            stats["total_swaps"],
            stats["num_layers"],
        ), stats

    if mpo_core is None or route_score == "none":
        return (stats["total_swaps"], stats["max_layer_swaps"], stats["num_layers"]), stats

    mpo_tmp = mpo_core.copy()
    peak_elems = elem_counts(mpo_tmp)
    peak_max_bond = mpo_tmp.max_bond()
    peak_bond_l2 = int(np.dot(get_bond_sizes(mpo_tmp), get_bond_sizes(mpo_tmp)))
    consumed_layers = 0

    body = layers[:-2] if len(layers) >= 2 else layers
    for layer in body[:lookahead_layers]:
        if side == "left":
            mpo_tmp = apply_circuit(
                mpo_tmp,
                q2c(layer.inverse()),
                side="right",
                max_bond=max_bond,
                cutoff=cutoff,
            )
        elif side == "right":
            mpo_tmp = apply_circuit(
                mpo_tmp,
                q2c(layer),
                side="left",
                max_bond=max_bond,
                cutoff=cutoff,
            )
        else:
            raise ValueError(f"unsupported rewire side: {side}")
        consumed_layers += 1
        peak_elems = max(peak_elems, elem_counts(mpo_tmp))
        peak_max_bond = max(peak_max_bond, mpo_tmp.max_bond())
        bonds = get_bond_sizes(mpo_tmp)
        peak_bond_l2 = max(peak_bond_l2, int(np.dot(bonds, bonds)))

    final_elems = elem_counts(mpo_tmp)
    final_max_bond = mpo_tmp.max_bond()
    stats.update({
        "lookahead_layers_scored": consumed_layers,
        "lookahead_final_elems": final_elems,
        "lookahead_final_max_bond": final_max_bond,
        "lookahead_peak_elems": peak_elems,
        "lookahead_peak_max_bond": peak_max_bond,
        "lookahead_peak_bond_l2": peak_bond_l2,
    })

    if route_score == "lookahead_total":
        score = (final_elems, peak_elems, final_max_bond, peak_max_bond)
    elif route_score == "lookahead_peak":
        score = (peak_elems, peak_max_bond, final_elems, final_max_bond)
    elif route_score == "lookahead_hot":
        score = (peak_max_bond, peak_bond_l2, peak_elems, final_elems)
    else:
        raise ValueError(f"unsupported route_score mode: {route_score}")

    return score + (
        stats["swap_only_layers"],
        stats["total_swaps"],
        stats["num_layers"],
    ), stats


def rewire_layers_scored(
    ls,
    perm,
    seed=None,
    sabre_trials=10000,
    sabre_heuristic="decay",
    route_candidates=1,
    route_seed_stride=1009,
    route_score="none",
    route_score_lookahead=8,
    mpo_core=None,
    side=None,
    q2c=None,
    max_bond=None,
    cutoff=0.0,
    parallel_route_candidates=False,
    route_candidate_workers=None,
):
    if route_candidates <= 1 or route_score == "none":
        started = time.perf_counter()
        return (
            rewire_layers(
                ls,
                perm,
                seed=seed,
                sabre_trials=sabre_trials,
                sabre_heuristic=sabre_heuristic,
            ),
            {
                "route_candidates": 1,
                "route_score": route_score,
                "route_score_time_s": 0.0,
                "route_chosen_seed": seed,
                "route_rewire_time_s": time.perf_counter() - started,
            },
        )

    started = time.perf_counter()
    candidate_seeds = _candidate_seeds(seed, route_candidates, route_seed_stride)

    def build_candidate(cand_seed):
        route_started = time.perf_counter()
        cand_layers = rewire_layers(
            ls,
            perm,
            seed=cand_seed,
            sabre_trials=sabre_trials,
            sabre_heuristic=sabre_heuristic,
        )
        route_time_s = time.perf_counter() - route_started
        score_started = time.perf_counter()
        score, score_stats = _score_routed_layers(
            cand_layers,
            mpo_core=mpo_core,
            side=side,
            q2c=q2c,
            max_bond=max_bond,
            cutoff=cutoff,
            lookahead_layers=route_score_lookahead,
            route_score=route_score,
        )
        score_time_s = time.perf_counter() - score_started
        return {
            "seed": cand_seed,
            "layers": cand_layers,
            "score": score,
            "score_stats": score_stats,
            "route_time_s": route_time_s,
            "score_time_s": score_time_s,
        }

    if parallel_route_candidates:
        workers = route_candidate_workers or route_candidates
        workers = max(1, min(int(workers), route_candidates))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            candidates = list(executor.map(build_candidate, candidate_seeds))
    else:
        workers = 1
        candidates = [build_candidate(cand_seed) for cand_seed in candidate_seeds]

    chosen = min(candidates, key=lambda item: item["score"])
    metadata = {
        "route_candidates": route_candidates,
        "route_score": route_score,
        "route_score_lookahead": route_score_lookahead,
        "parallel_route_candidates": bool(parallel_route_candidates),
        "route_candidate_workers": workers,
        "route_chosen_seed": chosen["seed"],
        "route_score_value": chosen["score"],
        "route_score_time_s": sum(item["score_time_s"] for item in candidates),
        "route_rewire_time_s": sum(item["route_time_s"] for item in candidates),
        "route_candidate_scores": [
            {
                "seed": item["seed"],
                "score": item["score"],
                **item["score_stats"],
            }
            for item in candidates
        ],
        "route_total_time_s": time.perf_counter() - started,
    }
    return chosen["layers"], metadata


# ------------------------------------------------------------------
#  Unswapping
# ------------------------------------------------------------------

def get_bond_sizes(mpo: MatrixProductOperator):
    return np.array([mpo.bond_size(ii,ii+1) for ii in range(len(mpo.sites) - 1)])


def count_work_ops(circuit):
    return count_work_ops_from_ops(circuit.count_ops())


def count_work_ops_from_ops(ops):
    return sum(
        count
        for op, count in ops.items()
        if op not in IGNORED_WORK_OPS
    )


def score_absorb_candidate(mpo: MatrixProductOperator, mode):
    if mode == "total_elems":
        return elem_counts(mpo)
    if mode == "max_bond":
        return (mpo.max_bond(), elem_counts(mpo))
    if mode == "bond_l2":
        bonds = get_bond_sizes(mpo)
        return (int(np.dot(bonds, bonds)), elem_counts(mpo))
    if mode == "hot_elems":
        bonds = get_bond_sizes(mpo)
        return (mpo.max_bond(), int(np.dot(bonds, bonds)), elem_counts(mpo))
    raise ValueError(f"unsupported absorb_score mode: {mode}")


def swap_perm(perm, swaps):
    for q0, q1 in swaps:
        (perm[q0], perm[q1]) = (perm[q1], perm[q0])
    return perm


def permutation_alignment_score(perm_left, perm_right):
    if len(perm_left) != len(perm_right):
        raise ValueError("permutations must have the same length")
    inv_left = np.argsort(perm_left)
    inv_right = np.argsort(perm_right)
    return int(np.abs(inv_left - inv_right).sum())


def alignment_delta_for_swaps(perm_left, perm_right, swaps, how):
    current_score = permutation_alignment_score(perm_left, perm_right)
    next_left = list(perm_left)
    next_right = list(perm_right)
    if how in ("left", "both"):
        next_left = swap_perm(next_left, list(swaps))
    if how in ("right", "both"):
        next_right = swap_perm(next_right, list(swaps))
    return permutation_alignment_score(next_left, next_right) - current_score


def layer_two_qubit_pairs(layer, include_swaps=False):
    pairs = []
    for instruction in layer.data:
        op_name = instruction.operation.name
        if op_name in {"measure", "barrier", "delay"}:
            continue
        if op_name == "swap" and not include_swaps:
            continue
        qargs = instruction.qubits
        if len(qargs) != 2:
            continue
        pairs.append((qargs[0]._index, qargs[1]._index))
    return pairs


def frontier_span_score(layers, perm, lookahead_layers=8, include_swaps=False):
    if not layers:
        return 0
    mapped_positions = np.argsort(perm)
    score = 0
    counted_layers = 0
    for layer in layers:
        layer_score = 0
        for q0, q1 in layer_two_qubit_pairs(layer, include_swaps=include_swaps):
            span = abs(int(mapped_positions[q0]) - int(mapped_positions[q1]))
            layer_score += max(0, span - 1)
        if layer_score:
            score += layer_score
            counted_layers += 1
            if counted_layers >= lookahead_layers:
                break
    return int(score)


def route_proxy_delta(
    future_layers_left,
    future_layers_right,
    perm_left,
    perm_right,
    swaps,
    how,
    lookahead_layers,
    include_swaps=False,
):
    current_score = 0
    next_score = 0
    if how in ("left", "both") and future_layers_left is not None:
        next_left = swap_perm(list(perm_left), list(swaps))
        current_score += frontier_span_score(
            future_layers_left,
            perm_left,
            lookahead_layers=lookahead_layers,
            include_swaps=include_swaps,
        )
        next_score += frontier_span_score(
            future_layers_left,
            next_left,
            lookahead_layers=lookahead_layers,
            include_swaps=include_swaps,
        )
    if how in ("right", "both") and future_layers_right is not None:
        next_right = swap_perm(list(perm_right), list(swaps))
        current_score += frontier_span_score(
            future_layers_right,
            perm_right,
            lookahead_layers=lookahead_layers,
            include_swaps=include_swaps,
        )
        next_score += frontier_span_score(
            future_layers_right,
            next_right,
            lookahead_layers=lookahead_layers,
            include_swaps=include_swaps,
        )
    return next_score - current_score


def select_route_proxy_swaps(
    candidate_ids,
    bond_gains,
    all_pairs,
    candidate_pairs,
    parity,
    how,
    perm_left,
    perm_right,
    future_layers_left,
    future_layers_right,
    route_proxy_weight,
    route_proxy_lookahead,
    route_proxy_include_swaps=False,
    route_proxy_policy="veto",
    route_proxy_min_benefit=1,
    route_proxy_max_bond_loss=0,
    route_proxy_protect_gain=None,
    alignment_weight=0.0,
):
    candidate_pair_set = set(candidate_pairs)
    improved_id_set = set(int(swap_id) for swap_id in np.asarray(candidate_ids).tolist())
    new_swaps = []
    rejected = 0
    added = 0
    route_proxy_weight = float(route_proxy_weight or 0.0)
    all_candidate_ids = [pair[0] for pair in candidate_pairs]
    if route_proxy_policy in ("augment", "hybrid"):
        ids_to_score = all_candidate_ids
    else:
        ids_to_score = list(candidate_ids)
    for swap_id in ids_to_score:
        if swap_id % 2 != parity:
            continue
        pair = all_pairs[swap_id]
        if pair not in candidate_pair_set:
            continue
        bond_gain = float(bond_gains[swap_id])
        is_improved = swap_id in improved_id_set
        if route_proxy_policy == "augment" and is_improved:
            new_swaps.append(pair)
            continue
        if (
            route_proxy_policy in ("veto", "hybrid")
            and is_improved
            and route_proxy_protect_gain is not None
            and bond_gain >= route_proxy_protect_gain
        ):
            new_swaps.append(pair)
            continue
        proxy_delta = route_proxy_delta(
            future_layers_left,
            future_layers_right,
            perm_left,
            perm_right,
            [pair],
            how,
            route_proxy_lookahead,
            include_swaps=route_proxy_include_swaps,
        )
        alignment_delta = alignment_delta_for_swaps(
            perm_left,
            perm_right,
            [pair],
            how,
        )
        score = (
            bond_gain
            - route_proxy_weight * proxy_delta
            - float(alignment_weight or 0.0) * alignment_delta
        )
        route_benefit = -proxy_delta
        if route_proxy_policy == "augment":
            if (
                not is_improved
                and route_benefit >= route_proxy_min_benefit
                and bond_gain >= -route_proxy_max_bond_loss
                and score > 0
            ):
                new_swaps.append(pair)
                added += 1
            else:
                rejected += 1
        elif route_proxy_policy == "hybrid":
            if score > 0:
                if not is_improved:
                    added += 1
                new_swaps.append(pair)
            else:
                rejected += 1
        elif score > 0:
            new_swaps.append(pair)
        else:
            rejected += 1
    return new_swaps, rejected, added


def select_aligned_swaps(
    candidate_ids,
    bond_gains,
    all_pairs,
    candidate_pairs,
    parity,
    how,
    perm_left,
    perm_right,
    alignment_weight,
    alignment_protect_gain=None,
):
    candidate_pair_set = set(candidate_pairs)
    new_swaps = []
    rejected = 0
    alignment_weight = float(alignment_weight or 0.0)
    for swap_id in np.asarray(candidate_ids).tolist():
        swap_id = int(swap_id)
        if swap_id % 2 != parity:
            continue
        pair = all_pairs[swap_id]
        if pair not in candidate_pair_set:
            continue
        bond_gain = float(bond_gains[swap_id])
        if alignment_protect_gain is not None and bond_gain >= alignment_protect_gain:
            new_swaps.append(pair)
            continue
        alignment_delta = alignment_delta_for_swaps(
            perm_left,
            perm_right,
            [pair],
            how,
        )
        if bond_gain - alignment_weight * alignment_delta > 0:
            new_swaps.append(pair)
        else:
            rejected += 1
    return new_swaps, rejected


def select_alignment_budget_swaps(
    baseline_ids,
    bond_gains,
    all_pairs,
    candidate_pairs,
    parity,
    how,
    perm_left,
    perm_right,
    alignment_weight,
    max_replacements=None,
):
    candidate_pair_set = set(candidate_pairs)
    baseline_ids = [
        int(swap_id)
        for swap_id in np.asarray(baseline_ids).tolist()
        if int(swap_id) % 2 == parity
        and all_pairs[int(swap_id)] in candidate_pair_set
    ]
    budget = len(baseline_ids)
    if budget == 0:
        return [], 0, 0
    baseline_id_set = set(baseline_ids)

    scored = []
    for pair in candidate_pairs:
        swap_id = pair[0]
        if swap_id % 2 != parity:
            continue
        bond_gain = float(bond_gains[swap_id])
        alignment_delta = alignment_delta_for_swaps(
            perm_left,
            perm_right,
            [pair],
            how,
        )
        score = bond_gain - float(alignment_weight or 0.0) * alignment_delta
        scored.append((score, bond_gain, -alignment_delta, swap_id, pair))

    scored = sorted(scored, reverse=True)
    if max_replacements is None or max_replacements < 0:
        chosen = scored[:budget]
    else:
        score_by_id = {swap_id: item for item in scored for swap_id in [item[3]]}
        chosen_by_id = {
            swap_id: score_by_id[swap_id]
            for swap_id in baseline_ids
            if swap_id in score_by_id
        }
        nonbaseline = [item for item in scored if item[3] not in baseline_id_set]
        replacements_left = int(max_replacements)
        for item in nonbaseline:
            if replacements_left <= 0:
                break
            if not chosen_by_id:
                break
            worst_baseline_id, worst_baseline_item = min(
                chosen_by_id.items(),
                key=lambda entry: entry[1],
            )
            if item <= worst_baseline_item:
                break
            chosen_by_id.pop(worst_baseline_id)
            chosen_by_id[item[3]] = item
            replacements_left -= 1
        chosen = sorted(chosen_by_id.values(), reverse=True)
    chosen_ids = {swap_id for _, _, _, swap_id, _ in chosen}
    replacements = len(chosen_ids - baseline_id_set)
    rejected = len(baseline_id_set - chosen_ids)
    return [pair for _, _, _, _, pair in chosen], rejected, replacements


def select_alignment_tiebreak_swaps(
    baseline_ids,
    bond_gains,
    all_pairs,
    candidate_pairs,
    parity,
    how,
    perm_left,
    perm_right,
    max_bond_gain_loss=1.0,
    max_replacements=1,
):
    candidate_pair_set = set(candidate_pairs)
    baseline_ids = [
        int(swap_id)
        for swap_id in np.asarray(baseline_ids).tolist()
        if int(swap_id) % 2 == parity
        and all_pairs[int(swap_id)] in candidate_pair_set
    ]
    if not baseline_ids:
        return [], 0, 0

    baseline_id_set = set(baseline_ids)

    def item_for(pair):
        swap_id = pair[0]
        bond_gain = float(bond_gains[swap_id])
        alignment_delta = alignment_delta_for_swaps(
            perm_left,
            perm_right,
            [pair],
            how,
        )
        return {
            "swap_id": swap_id,
            "pair": pair,
            "bond_gain": bond_gain,
            "alignment_delta": alignment_delta,
        }

    selected = {
        swap_id: item_for(all_pairs[swap_id])
        for swap_id in baseline_ids
    }
    replacements_left = int(max_replacements if max_replacements is not None else 1)
    candidates = [
        item_for(pair)
        for pair in candidate_pairs
        if pair[0] % 2 == parity and pair[0] not in baseline_id_set
    ]
    candidates.sort(
        key=lambda item: (
            item["alignment_delta"],
            -item["bond_gain"],
            item["swap_id"],
        )
    )
    for candidate in candidates:
        if replacements_left <= 0:
            break
        eligible = [
            item
            for item in selected.values()
            if (
                item["bond_gain"] - candidate["bond_gain"] <= max_bond_gain_loss
                and candidate["alignment_delta"] < item["alignment_delta"]
            )
        ]
        if not eligible:
            continue
        victim = max(
            eligible,
            key=lambda item: (
                item["alignment_delta"] - candidate["alignment_delta"],
                candidate["bond_gain"] - item["bond_gain"],
            ),
        )
        selected.pop(victim["swap_id"])
        selected[candidate["swap_id"]] = candidate
        replacements_left -= 1

    chosen_ids = set(selected)
    replacements = len(chosen_ids - baseline_id_set)
    rejected = len(baseline_id_set - chosen_ids)
    return [item["pair"] for item in selected.values()], rejected, replacements


def get_good_swaps(
    mpo,
    qubit_pairs,
    how,
    max_bond,
    cutoff,
    to_backend=None,
    equal=False,
    probe_max_bond=None,
    probe_cutoff=None,
    swap_apply_method="mpo",
    swap_gate_representation="cx",
    return_mpo=False,
    return_gains=False,
):
    current_bonds = get_bond_sizes(mpo)
    #log_print("    [debug](select)(bond sizes before) -> ", current_bonds.tolist())

    swaps_l = qubit_pairs if how in ("left", "both") else []
    swaps_r = qubit_pairs if how in ("right", "both") else []
    selection_max_bond = probe_max_bond if probe_max_bond is not None else max_bond
    selection_cutoff = probe_cutoff if probe_cutoff is not None else cutoff

    mpo_tmp = apply_swaps(
        mpo,
        swaps_l=swaps_l,
        swaps_r=swaps_r,
        max_bond=selection_max_bond,
        cutoff=selection_cutoff,
        to_backend=to_backend,
        method=swap_apply_method,
        swap_gate_representation=swap_gate_representation,
    )
    new_bonds = get_bond_sizes(mpo_tmp)
    if equal is None:
        new_bonds = new_bonds + (np.random.rand(*new_bonds.shape)-0.5)
        improved = np.nonzero(new_bonds < current_bonds)[0]
    elif equal:
        improved = np.nonzero(new_bonds <= current_bonds)[0]
    else:
        improved = np.nonzero(new_bonds < current_bonds)[0]

    if return_mpo and return_gains:
        return improved, mpo_tmp, current_bonds - new_bonds
    if return_gains:
        return improved, current_bonds - new_bonds
    if return_mpo:
        return improved, mpo_tmp
    return improved


def filter_hot_pairs(mpo, candidate_pairs, hot_bonds=0, hot_radius=0):
    if hot_bonds is None or hot_bonds <= 0:
        return candidate_pairs
    if not candidate_pairs:
        return candidate_pairs

    bonds = get_bond_sizes(mpo)
    if len(bonds) == 0:
        return candidate_pairs

    top_k = min(int(hot_bonds), len(bonds))
    hot_indices = np.argsort(bonds)[-top_k:]
    hot_set = {
        idx + offset
        for idx in hot_indices
        for offset in range(-int(hot_radius), int(hot_radius) + 1)
        if 0 <= idx + offset < len(bonds)
    }
    return [pair for pair in candidate_pairs if pair[0] in hot_set]


def probe_swap_layer(
    mpo,
    qubit_pairs,
    how,
    max_bond,
    cutoff,
    to_backend=None,
    swap_apply_method="mpo",
    swap_gate_representation="cx",
):
    swaps_l = qubit_pairs if how in ("left", "both") else []
    swaps_r = qubit_pairs if how in ("right", "both") else []
    return apply_swaps(
        mpo,
        swaps_l=swaps_l,
        swaps_r=swaps_r,
        max_bond=max_bond,
        cutoff=cutoff,
        to_backend=to_backend,
        method=swap_apply_method,
        swap_gate_representation=swap_gate_representation,
    )


def limit_pairs_by_hot_bonds(mpo, candidate_pairs, limit):
    if limit is None or limit <= 0 or len(candidate_pairs) <= limit:
        return candidate_pairs
    bonds = get_bond_sizes(mpo)
    if len(bonds) == 0:
        return candidate_pairs[:limit]
    return sorted(
        candidate_pairs,
        key=lambda pair: bonds[pair[0]] if pair[0] < len(bonds) else 0,
        reverse=True,
    )[:limit]


def choose_pair_lookahead_swaps(
    mpo,
    candidate_pairs,
    how,
    max_bond,
    cutoff,
    to_backend=None,
    swap_apply_method="mpo",
    swap_gate_representation="cx",
    pair_limit=8,
):
    limited_pairs = limit_pairs_by_hot_bonds(mpo, candidate_pairs, pair_limit)
    current_score = (elem_counts(mpo), mpo.max_bond())
    best_score = current_score
    best_pairs = []
    best_mpo = mpo
    probes = 0

    pair_sets = [[pair] for pair in limited_pairs]
    for idx, first_pair in enumerate(limited_pairs):
        for second_pair in limited_pairs[idx + 1:]:
            pair_sets.append([first_pair, second_pair])

    for pair_set in pair_sets:
        candidate_mpo = probe_swap_layer(
            mpo,
            pair_set,
            how=how,
            max_bond=max_bond,
            cutoff=cutoff,
            to_backend=to_backend,
            swap_apply_method=swap_apply_method,
            swap_gate_representation=swap_gate_representation,
        )
        probes += 1
        candidate_score = (elem_counts(candidate_mpo), candidate_mpo.max_bond())
        if candidate_score < best_score:
            best_score = candidate_score
            best_pairs = pair_set
            best_mpo = candidate_mpo

    return best_pairs, best_mpo, probes, len(limited_pairs), current_score, best_score


def unswap(
    mpo: MatrixProductOperator,
    hows=("left", "right", "both"),
    max_bond=2048,
    cutoff=0.0001,
    max_its=25,
    equal=False,
    to_backend=None,
    t0=0,
    stop_total_elems=None,
    stop_max_bond=None,
    probe_max_bond=None,
    probe_cutoff=None,
    adaptive_stop_max_bond=None,
    adaptive_stop_min_rel_improvement=None,
    adaptive_stop_min_iteration=0,
    swap_apply_method="mpo",
    swap_gate_representation="cx",
    unswap_select_mode="bond",
    reuse_full_swap_probe=False,
    unswap_hot_bonds=0,
    unswap_hot_radius=0,
    unswap_pair_lookahead_limit=8,
    future_layers_left=None,
    future_layers_right=None,
    unswap_route_proxy_weight=0.0,
    unswap_route_proxy_lookahead=8,
    unswap_route_proxy_include_swaps=False,
    unswap_route_proxy_allow_nonbond=False,
    unswap_route_proxy_policy="veto",
    unswap_route_proxy_min_benefit=1,
    unswap_route_proxy_max_bond_loss=0,
    unswap_route_proxy_protect_gain=None,
    unswap_alignment_weight=0.0,
    unswap_alignment_protect_gain=None,
    unswap_alignment_max_replacements=None,
    unswap_alignment_tie_loss=1.0,
):
    num_qubits = len(mpo.sites)
    all_pairs = [(i, i+1) for i in range(num_qubits-1)]

    perm_left = list(range(len(mpo.sites)))
    perm_right = list(range(len(mpo.sites)))

    logging.info("    [start unswap] -> " + str(get_tn_info(mpo)))
    num_improvements = 1
    start_counts = 1
    end_counts = 0
    ii = 0

    stats_data = []
    while num_improvements > 0 and ii < max_its and start_counts != end_counts:
        num_improvements = 0
        start_counts = elem_counts(mpo)

        for how in hows:
            for parity in [0, 1]:
                all_candidate_pairs = all_pairs[parity::2]
                candidate_pairs = filter_hot_pairs(
                    mpo,
                    all_candidate_pairs,
                    hot_bonds=unswap_hot_bonds,
                    hot_radius=unswap_hot_radius,
                )
                if not candidate_pairs:
                    new_swaps = []
                    stats_data.append({
                        "time": time.perf_counter() - t0,
                        "stage": "unswapping",
                        "iteration": ii,
                        "side": how,
                        "parity": parity,
                        "candidate_pairs": 0,
                        "new_swaps": 0,
                        "total_swaps": num_improvements,
                        "alignment_score": permutation_alignment_score(perm_left, perm_right),
                        **get_tn_info(mpo),
                    })
                    logging.info(
                        f"    [{ii} | {how} | {parity}]"
                        f"(candidate_pairs: 0 | new_swaps: 0 | total: {num_improvements}) -> "
                        + str(get_tn_info(mpo))
                    )
                    continue
                route_proxy_rejected_swaps = None
                route_proxy_added_swaps = None
                if unswap_select_mode in ("bond", "bond_aligned", "bond_aligned_budget", "bond_aligned_tiebreak", "bond_route_proxy"):
                    # Estimate which qubit pairs to swap.
                    if reuse_full_swap_probe:
                        if unswap_select_mode in ("bond_aligned", "bond_aligned_budget", "bond_aligned_tiebreak", "bond_route_proxy"):
                            new_swap_ids, full_probe_mpo, bond_gains = get_good_swaps(
                                mpo,
                                qubit_pairs=candidate_pairs,
                                how=how,
                                max_bond=max_bond,
                                cutoff=cutoff,
                                to_backend=to_backend,
                                equal=equal,
                                probe_max_bond=probe_max_bond,
                                probe_cutoff=probe_cutoff,
                                swap_apply_method=swap_apply_method,
                                swap_gate_representation=swap_gate_representation,
                                return_mpo=True,
                                return_gains=True,
                            )
                        else:
                            new_swap_ids, full_probe_mpo = get_good_swaps(
                                mpo,
                                qubit_pairs=candidate_pairs,
                                how=how,
                                max_bond=max_bond,
                                cutoff=cutoff,
                                to_backend=to_backend,
                                equal=equal,
                                probe_max_bond=probe_max_bond,
                                probe_cutoff=probe_cutoff,
                                swap_apply_method=swap_apply_method,
                                swap_gate_representation=swap_gate_representation,
                                return_mpo=True,
                            )
                            bond_gains = None
                    else:
                        if unswap_select_mode in ("bond_aligned", "bond_aligned_budget", "bond_aligned_tiebreak", "bond_route_proxy"):
                            new_swap_ids, bond_gains = get_good_swaps(
                                mpo,
                                qubit_pairs=candidate_pairs,
                                how=how,
                                max_bond=max_bond,
                                cutoff=cutoff,
                                to_backend=to_backend,
                                equal=equal,
                                probe_max_bond=probe_max_bond,
                                probe_cutoff=probe_cutoff,
                                swap_apply_method=swap_apply_method,
                                swap_gate_representation=swap_gate_representation,
                                return_gains=True,
                            )
                        else:
                            new_swap_ids = get_good_swaps(
                                mpo,
                                qubit_pairs=candidate_pairs,
                                how=how,
                                max_bond=max_bond,
                                cutoff=cutoff,
                                to_backend=to_backend,
                                equal=equal,
                                probe_max_bond=probe_max_bond,
                                probe_cutoff=probe_cutoff,
                                swap_apply_method=swap_apply_method,
                                swap_gate_representation=swap_gate_representation,
                            )
                            bond_gains = None
                        full_probe_mpo = None
                    if unswap_select_mode == "bond_route_proxy":
                        if unswap_route_proxy_allow_nonbond:
                            route_proxy_candidate_ids = np.array(
                                [pair[0] for pair in candidate_pairs],
                                dtype=int,
                            )
                        else:
                            route_proxy_candidate_ids = new_swap_ids
                        new_swaps, route_proxy_rejected_swaps, route_proxy_added_swaps = select_route_proxy_swaps(
                            route_proxy_candidate_ids,
                            bond_gains,
                            all_pairs,
                            candidate_pairs,
                            parity,
                            how,
                            perm_left,
                            perm_right,
                            future_layers_left,
                            future_layers_right,
                            unswap_route_proxy_weight,
                            unswap_route_proxy_lookahead,
                            route_proxy_include_swaps=unswap_route_proxy_include_swaps,
                            route_proxy_policy=unswap_route_proxy_policy,
                            route_proxy_min_benefit=unswap_route_proxy_min_benefit,
                            route_proxy_max_bond_loss=unswap_route_proxy_max_bond_loss,
                            route_proxy_protect_gain=unswap_route_proxy_protect_gain,
                            alignment_weight=unswap_alignment_weight,
                        )
                    elif unswap_select_mode == "bond_aligned":
                        new_swaps, route_proxy_rejected_swaps = select_aligned_swaps(
                            new_swap_ids,
                            bond_gains,
                            all_pairs,
                            candidate_pairs,
                            parity,
                            how,
                            perm_left,
                            perm_right,
                            unswap_alignment_weight,
                            alignment_protect_gain=unswap_alignment_protect_gain,
                        )
                    elif unswap_select_mode == "bond_aligned_budget":
                        new_swaps, route_proxy_rejected_swaps, route_proxy_added_swaps = select_alignment_budget_swaps(
                            new_swap_ids,
                            bond_gains,
                            all_pairs,
                            candidate_pairs,
                            parity,
                            how,
                            perm_left,
                            perm_right,
                            unswap_alignment_weight,
                            max_replacements=unswap_alignment_max_replacements,
                        )
                    elif unswap_select_mode == "bond_aligned_tiebreak":
                        new_swaps, route_proxy_rejected_swaps, route_proxy_added_swaps = select_alignment_tiebreak_swaps(
                            new_swap_ids,
                            bond_gains,
                            all_pairs,
                            candidate_pairs,
                            parity,
                            how,
                            perm_left,
                            perm_right,
                            max_bond_gain_loss=unswap_alignment_tie_loss,
                            max_replacements=unswap_alignment_max_replacements,
                        )
                    else:
                        candidate_pair_set = set(candidate_pairs)
                        new_swaps = [
                            all_pairs[i]
                            for i in new_swap_ids
                            if i % 2 == parity and all_pairs[i] in candidate_pair_set
                        ]

                    # Apply the selected swaps.
                    swaps_l = new_swaps if how in ("left", "both") else []
                    swaps_r = new_swaps if how in ("right", "both") else []
                    if new_swaps:
                        can_reuse_full_probe = (
                            reuse_full_swap_probe
                            and
                            probe_max_bond is None
                            and probe_cutoff is None
                            and len(new_swaps) == len(candidate_pairs)
                        )
                        if can_reuse_full_probe:
                            mpo = full_probe_mpo
                        else:
                            mpo = apply_swaps(
                                mpo,
                                swaps_l=swaps_l,
                                swaps_r=swaps_r,
                                max_bond=max_bond,
                                cutoff=cutoff,
                                to_backend=to_backend,
                                method=swap_apply_method,
                                swap_gate_representation=swap_gate_representation,
                            )
                elif unswap_select_mode == "layer":
                    candidate_mpo = probe_swap_layer(
                        mpo,
                        candidate_pairs,
                        how=how,
                        max_bond=probe_max_bond if probe_max_bond is not None else max_bond,
                        cutoff=probe_cutoff if probe_cutoff is not None else cutoff,
                        to_backend=to_backend,
                        swap_apply_method=swap_apply_method,
                        swap_gate_representation=swap_gate_representation,
                    )
                    candidate_score = (
                        elem_counts(candidate_mpo),
                        candidate_mpo.max_bond(),
                    )
                    current_score = (elem_counts(mpo), mpo.max_bond())
                    if candidate_score < current_score:
                        mpo = candidate_mpo
                        new_swaps = candidate_pairs
                    else:
                        new_swaps = []
                elif unswap_select_mode == "pair_lookahead":
                    (
                        new_swaps,
                        candidate_mpo,
                        pair_lookahead_probes,
                        pair_lookahead_candidates,
                        pair_lookahead_current_score,
                        pair_lookahead_best_score,
                    ) = choose_pair_lookahead_swaps(
                        mpo,
                        candidate_pairs,
                        how=how,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        to_backend=to_backend,
                        swap_apply_method=swap_apply_method,
                        swap_gate_representation=swap_gate_representation,
                        pair_limit=unswap_pair_lookahead_limit,
                    )
                    if new_swaps:
                        mpo = candidate_mpo
                else:
                    raise ValueError(f"unsupported unswap_select_mode: {unswap_select_mode}")

                # Update the permutations
                if new_swaps:
                    if how in ("left", "both"):
                        perm_left = swap_perm(perm_left, new_swaps)
                    if how in ("right", "both"):
                        perm_right = swap_perm(perm_right, new_swaps)
    
                # Track how many new swaps were applied
                num_improvements += len(new_swaps)
                stats_data.append({
                    "time": time.perf_counter() - t0,
                    "stage": "unswapping",
                    "iteration": ii,
                    "side": how,
                    "parity": parity,
                    "candidate_pairs": len(candidate_pairs),
                    "candidate_pairs_total": len(all_candidate_pairs),
                    "new_swaps": len(new_swaps),
                    "total_swaps": num_improvements,
                    "pair_lookahead_probes": pair_lookahead_probes if unswap_select_mode == "pair_lookahead" else None,
                    "pair_lookahead_candidates": pair_lookahead_candidates if unswap_select_mode == "pair_lookahead" else None,
                    "pair_lookahead_current_score": pair_lookahead_current_score if unswap_select_mode == "pair_lookahead" else None,
                    "pair_lookahead_best_score": pair_lookahead_best_score if unswap_select_mode == "pair_lookahead" else None,
                    "alignment_score": permutation_alignment_score(perm_left, perm_right),
                    "route_proxy_weight": unswap_route_proxy_weight if unswap_select_mode == "bond_route_proxy" else None,
                    "route_proxy_lookahead": unswap_route_proxy_lookahead if unswap_select_mode == "bond_route_proxy" else None,
                    "route_proxy_rejected_swaps": route_proxy_rejected_swaps,
                    "route_proxy_added_swaps": route_proxy_added_swaps,
                    "alignment_weight": unswap_alignment_weight if unswap_select_mode in ("bond_aligned", "bond_aligned_budget", "bond_route_proxy") else None,
                    "alignment_protect_gain": unswap_alignment_protect_gain if unswap_select_mode == "bond_aligned" else None,
                    "alignment_tie_loss": unswap_alignment_tie_loss if unswap_select_mode == "bond_aligned_tiebreak" else None,
                    **get_tn_info(mpo),
                })
                logging.info(
                    f"    [{ii} | {how} | {parity}]"
                    f"(candidate_pairs: {len(candidate_pairs)}/{len(all_candidate_pairs)} | "
                    f"new_swaps: {len(new_swaps)} | total: {num_improvements}) -> "
                    + str(get_tn_info(mpo))
                )
                small_enough = (
                    (stop_total_elems is None or elem_counts(mpo) <= stop_total_elems)
                    and (stop_max_bond is None or mpo.max_bond() <= stop_max_bond)
                )
                if (stop_total_elems is not None or stop_max_bond is not None) and small_enough:
                    logging.info(f"    [end unswap: stop target] -> " + str(get_tn_info(mpo)))
                    return mpo, (perm_left, perm_right), stats_data

        end_counts = elem_counts(mpo)
        ii += 1
        if (
            adaptive_stop_max_bond is not None
            and adaptive_stop_min_rel_improvement is not None
            and ii >= adaptive_stop_min_iteration
            and mpo.max_bond() <= adaptive_stop_max_bond
        ):
            rel_improvement = (
                (start_counts - end_counts) / start_counts
                if start_counts > 0
                else 0.0
            )
            if rel_improvement < adaptive_stop_min_rel_improvement:
                logging.info(
                    "    [end unswap: adaptive stop]"
                    f"(rel_improvement: {rel_improvement:.6g}) -> "
                    + str(get_tn_info(mpo))
                )
                stats_data.append({
                    "time": time.perf_counter() - t0,
                    "stage": "unswap_adaptive_stop",
                    "iteration": ii,
                    "rel_improvement": rel_improvement,
                    "adaptive_stop_max_bond": adaptive_stop_max_bond,
                    "adaptive_stop_min_rel_improvement": adaptive_stop_min_rel_improvement,
                    **get_tn_info(mpo),
                })
                break
    logging.info(f"    [end unswap] -> " + str(get_tn_info(mpo)))

    return mpo, (perm_left, perm_right), stats_data


# ------------------------------------------------------------------
#  MPO Cancellation + Unswapping
# ------------------------------------------------------------------

def mpo_compress_unswap(
    circuit: QuantumCircuit,
    max_bond=8192,
    cutoff=0.001,
    unswap_threshold=1e6,
    early_stopping_gates=100,
    center_ratio=0.5,
    equal=False,
    flip_freq=None,
    max_its=20,
    to_backend=None,
    seed=None,
    hows=("both", "left", "right"),
    mpo_core=None,
    sabre_trials=10000,
    post_sabre_trials=None,
    post_sabre_seed=None,
    sabre_heuristic="decay",
    on_stats=None,
    max_unswap_cycles=None,
    max_work_gates=None,
    abort_after_no_progress_unswap_cycles=2,
    unswap_stop_total_elems=None,
    unswap_stop_max_bond=None,
    unswap_probe_max_bond=None,
    unswap_probe_cutoff=None,
    absorb_score="total_elems",
    parallel_absorb_probes=False,
    parallel_rewire=False,
    adaptive_parallel_rewire=False,
    adaptive_parallel_rewire_min_speedup=1.15,
    unswap_adaptive_stop_max_bond=None,
    unswap_adaptive_stop_min_rel_improvement=None,
    unswap_adaptive_stop_min_iteration=0,
    route_candidates=1,
    route_seed_stride=1009,
    route_score="none",
    route_score_lookahead=8,
    parallel_route_candidates=False,
    route_candidate_workers=None,
    swap_apply_method="mpo",
    swap_gate_representation="current",
    unswap_select_mode="bond",
    reuse_full_swap_probe=False,
    unswap_trigger_max_bond=None,
    unswap_hot_bonds=0,
    unswap_hot_radius=0,
    absorb_lookahead_depth=1,
    unswap_pair_lookahead_limit=8,
    unswap_route_proxy_weight=0.0,
    unswap_route_proxy_lookahead=8,
    unswap_route_proxy_include_swaps=False,
    unswap_route_proxy_allow_nonbond=False,
    unswap_route_proxy_policy="veto",
    unswap_route_proxy_min_benefit=1,
    unswap_route_proxy_max_bond_loss=0,
    unswap_route_proxy_protect_gain=None,
    unswap_route_proxy_max_cycles=None,
    unswap_alignment_weight=0.0,
    unswap_alignment_protect_gain=None,
    unswap_alignment_max_replacements=None,
    unswap_alignment_tie_loss=1.0,
):
    if swap_gate_representation == "current":
        routed_swap_representation = "block"
        unswap_swap_representation = "cx"
    elif swap_gate_representation in ("block", "cx"):
        routed_swap_representation = swap_gate_representation
        unswap_swap_representation = swap_gate_representation
    else:
        raise ValueError(
            f"unsupported SWAP gate representation: {swap_gate_representation}"
        )

    def q2c(qc):
        qc = qc.decompose("unitary")
        if routed_swap_representation == "cx":
            qc = qc.decompose("swap")
        return quimb_circuit(qc, Circuit, to_backend=to_backend)

    t0 = time.perf_counter()

    # Split circuit into left and right
    if type(center_ratio) is float:
        C = int(len(circuit) * center_ratio)
    elif type(center_ratio) is int:
        C = center_ratio
    circuit_left = merge_gates(circuit[:C], circuit.num_qubits).inverse()
    circuit_right = merge_gates(circuit[C:], circuit.num_qubits)
    if "measure" not in circuit_right.count_ops():
        circuit_right.measure_all()
    if "measure" not in circuit_left.count_ops():
        circuit_left.measure_all()

    layers_left = list(iter_layers(circuit_left))
    layers_right = list(iter_layers(circuit_right))


    T_U = count_work_ops(circuit)
    T_UL = count_work_ops(circuit_left)
    T_UR = count_work_ops(circuit_right)

    logging.info(f"Total unitaries: {T_U} = {T_UL} (left) + {T_UR} (right)")

    # Rewire layers
    logging.info(
        f"Rewiring with SabreSwap(trials={sabre_trials}, "
        f"heuristic={sabre_heuristic!r}, seed={seed})"
    )
    if route_candidates > 1 and route_score != "none":
        logging.info(
            "Post-unswap rewire candidate scoring enabled: "
            f"candidates={route_candidates}, score={route_score!r}, "
            f"lookahead={route_score_lookahead}, seed_stride={route_seed_stride}"
        )
    if (
        abort_after_no_progress_unswap_cycles is not None
        and abort_after_no_progress_unswap_cycles >= 0
    ):
        logging.info(
            "No-progress fail-fast guardrail enabled: abort after %s "
            "consecutive zero-work unswap cycle(s)",
            abort_after_no_progress_unswap_cycles,
        )
    post_rewire_sabre_trials = (
        sabre_trials if post_sabre_trials is None else post_sabre_trials
    )
    post_rewire_seed = seed if post_sabre_seed is None else post_sabre_seed

    stats_data = []
    timing_totals = {
        "initial_rewire_time_s": 0.0,
        "initial_rewire_wall_time_s": 0.0,
        "post_unswap_rewire_time_s": 0.0,
        "post_unswap_rewire_wall_time_s": 0.0,
        "absorb_probe_time_s": 0.0,
        "absorb_probe_wall_time_s": 0.0,
        "absorb_lookahead_time_s": 0.0,
        "unswap_time_s": 0.0,
    }
    adaptive_parallel_rewire_active = bool(parallel_rewire)
    adaptive_parallel_rewire_probed = False
    rewire_executor = (
        ThreadPoolExecutor(max_workers=2)
        if parallel_rewire or adaptive_parallel_rewire
        else None
    )

    def record_parallel_rewire_probe(phase, cycle, sequential_wall_s, parallel_wall_s):
        nonlocal adaptive_parallel_rewire_active, adaptive_parallel_rewire_probed
        speedup = (
            sequential_wall_s / parallel_wall_s
            if parallel_wall_s > 0
            else float("inf")
        )
        enabled = speedup >= adaptive_parallel_rewire_min_speedup
        adaptive_parallel_rewire_active = enabled
        adaptive_parallel_rewire_probed = True
        row = {
            "time": time.perf_counter() - t0,
            "stage": "rewire_parallel_probe",
            "rewire_phase": phase,
            "unswap_cycle": cycle,
            "sequential_rewire_wall_time_s": sequential_wall_s,
            "parallel_rewire_wall_time_s": parallel_wall_s,
            "parallel_rewire_probe_speedup": speedup,
            "adaptive_parallel_rewire_min_speedup": adaptive_parallel_rewire_min_speedup,
            "adaptive_parallel_rewire_enabled": enabled,
        }
        stats_data.append(row)
        if on_stats is not None:
            on_stats(row)
        logging.info(
            "[parallel rewire probe](phase=%s, cycle=%s, sequential_wall_s=%.2f, "
            "parallel_wall_s=%.2f, speedup=%.2f, enabled=%s)",
            phase,
            cycle,
            sequential_wall_s,
            parallel_wall_s,
            speedup,
            enabled,
        )

    def rewire_initial(side, layers):
        logging.info(
            "[start rewire](phase=initial, side=%s, layers=%s, trials=%s, seed=%s, parallel=%s)",
            side,
            len(layers),
            sabre_trials,
            seed,
            parallel_rewire,
        )
        rewire_started = time.perf_counter()
        routed_layers = rewire_layers(
            layers,
            np.arange(circuit.num_qubits, dtype=int),
            seed=seed,
            sabre_trials=sabre_trials,
            sabre_heuristic=sabre_heuristic,
        )
        rewire_time_s = time.perf_counter() - rewire_started
        logging.info(
            "[end rewire](phase=initial, side=%s, routed_layers=%s, elapsed_s=%.2f)",
            side,
            len(routed_layers),
            rewire_time_s,
        )
        return routed_layers, rewire_time_s, {
            "time": time.perf_counter() - t0,
            "stage": "rewiring",
            "rewire_side": side,
            "rewire_phase": "initial",
            "rewire_time_s": rewire_time_s,
            "remaining_layers_before_rewire": len(routed_layers),
            "parallel_rewire": parallel_rewire,
            "adaptive_parallel_rewire": adaptive_parallel_rewire,
        }

    initial_rewire_wall_started = time.perf_counter()
    if adaptive_parallel_rewire:
        left_input = layers_left
        right_input = layers_right
        layers_left, rewire_time_s, row = rewire_initial("left", left_input)
        timing_totals["initial_rewire_time_s"] += rewire_time_s
        stats_data.append(row)
        layers_right, rewire_time_s, row = rewire_initial("right", right_input)
        timing_totals["initial_rewire_time_s"] += rewire_time_s
        stats_data.append(row)
        sequential_initial_wall_s = time.perf_counter() - initial_rewire_wall_started

        probe_started = time.perf_counter()
        left_future = rewire_executor.submit(rewire_initial, "left", left_input)
        right_future = rewire_executor.submit(rewire_initial, "right", right_input)
        left_future.result()
        right_future.result()
        parallel_initial_wall_s = time.perf_counter() - probe_started
        record_parallel_rewire_probe(
            "initial",
            None,
            sequential_initial_wall_s,
            parallel_initial_wall_s,
        )
    elif parallel_rewire:
        left_future = rewire_executor.submit(rewire_initial, "left", layers_left)
        right_future = rewire_executor.submit(rewire_initial, "right", layers_right)
        layers_left, rewire_time_s, row = left_future.result()
        timing_totals["initial_rewire_time_s"] += rewire_time_s
        stats_data.append(row)
        layers_right, right_rewire_time_s, right_row = right_future.result()
        timing_totals["initial_rewire_time_s"] += right_rewire_time_s
        stats_data.append(right_row)
    else:
        layers_left, rewire_time_s, row = rewire_initial("left", layers_left)
        timing_totals["initial_rewire_time_s"] += rewire_time_s
        stats_data.append(row)
        layers_right, rewire_time_s, row = rewire_initial("right", layers_right)
        timing_totals["initial_rewire_time_s"] += rewire_time_s
        stats_data.append(row)
    timing_totals["initial_rewire_wall_time_s"] += time.perf_counter() - initial_rewire_wall_started
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    # Start the MPO and counters
    ii_left = 0
    ii_right = 0
    do_left = False
    if mpo_core is None:
        mpo_core = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    logging.info("[start compressing] -> " + str(get_tn_info(mpo_core)))


    total_u_consumed = 0
    current_u_consumed = 0
    total_u_consumed_left = 0
    total_u_consumed_right = 0

    unswap_cycles = 0
    no_progress_unswap_cycles = 0
    cycle_start_total_elems = elem_counts(mpo_core)
    termination_reason = "completed"
    termination_detail = None

    # Start loop
    probe_executor = ThreadPoolExecutor(max_workers=2) if parallel_absorb_probes else None
    while ii_left < len(layers_left) or ii_right < len(layers_right):
        # Try both sides to see which one results in a smaller size
        def absorb_layer(mpo, side, left_index, right_index):
            if side == "L":
                if left_index >= len(layers_left):
                    return None
                return apply_circuit(
                    mpo,
                    q2c(layers_left[left_index].inverse()),
                    side="right",
                    max_bond=max_bond,
                    cutoff=cutoff,
                )
            if right_index >= len(layers_right):
                return None
            return apply_circuit(
                mpo,
                q2c(layers_right[right_index]),
                side="left",
                max_bond=max_bond,
                cutoff=cutoff,
            )

        def is_absorbable(mpo):
            if mpo is None:
                return False
            if elem_counts(mpo) >= unswap_threshold:
                return False
            return unswap_trigger_max_bond is None or mpo.max_bond() <= unswap_trigger_max_bond

        def probe_left():
            try:
                probe_started = time.perf_counter()
                return (
                    absorb_layer(mpo_core, "L", ii_left, ii_right),
                    time.perf_counter() - probe_started,
                )
            except KeyboardInterrupt:
                raise

        def probe_right():
            try:
                probe_started = time.perf_counter()
                return (
                    absorb_layer(mpo_core, "R", ii_left, ii_right),
                    time.perf_counter() - probe_started,
                )
            except KeyboardInterrupt:
                raise

        probe_wall_started = time.perf_counter()
        if parallel_absorb_probes and ii_left < len(layers_left) and ii_right < len(layers_right):
            left_future = probe_executor.submit(probe_left)
            right_future = probe_executor.submit(probe_right)
            mpo_left, probe_left_time_s = left_future.result()
            mpo_right, probe_right_time_s = right_future.result()
            counts_left = elem_counts(mpo_left)
            counts_right = elem_counts(mpo_right)
            max_bond_left = mpo_left.max_bond()
            max_bond_right = mpo_right.max_bond()
        else:
            if ii_left < len(layers_left):
                mpo_left, probe_left_time_s = probe_left()
                counts_left = elem_counts(mpo_left)
                max_bond_left = mpo_left.max_bond()
            else:
                mpo_left = None
                counts_left = 1e20
                max_bond_left = 1e20
                probe_left_time_s = 0.0

            if ii_right < len(layers_right):
                mpo_right, probe_right_time_s = probe_right()
                counts_right = elem_counts(mpo_right)
                max_bond_right = mpo_right.max_bond()
            else:
                mpo_right = None
                counts_right = 1e20
                max_bond_right = 1e20
                probe_right_time_s = 0.0
        probe_wall_time_s = time.perf_counter() - probe_wall_started

        timing_totals["absorb_probe_time_s"] += probe_left_time_s + probe_right_time_s
        timing_totals["absorb_probe_wall_time_s"] += probe_wall_time_s
        
        lookahead_depth_used = 1
        lookahead_candidates = None
        lookahead_choice = None
        lookahead_score = None
        lookahead_time_s = 0.0
        if flip_freq is None:
            eligible = []
            left_absorbable = is_absorbable(mpo_left)
            right_absorbable = is_absorbable(mpo_right)
            if left_absorbable:
                eligible.append(("L", score_absorb_candidate(mpo_left, absorb_score)))
            if right_absorbable:
                eligible.append(("R", score_absorb_candidate(mpo_right, absorb_score)))
            if eligible:
                if absorb_lookahead_depth > 1:
                    lookahead_started = time.perf_counter()

                    def extend_absorb_path(mpo, left_offset, right_offset, depth_left):
                        if depth_left == 0:
                            return [(score_absorb_candidate(mpo, absorb_score), 0)]

                        candidates = []
                        next_left = ii_left + left_offset
                        next_right = ii_right + right_offset
                        for next_side in ("L", "R"):
                            next_mpo = absorb_layer(mpo, next_side, next_left, next_right)
                            if not is_absorbable(next_mpo):
                                continue
                            next_left_offset = left_offset + int(next_side == "L")
                            next_right_offset = right_offset + int(next_side == "R")
                            for score, consumed in extend_absorb_path(
                                next_mpo,
                                next_left_offset,
                                next_right_offset,
                                depth_left - 1,
                            ):
                                candidates.append((score, consumed + 1))
                        if not candidates:
                            return [(score_absorb_candidate(mpo, absorb_score), 0)]
                        return candidates

                    lookahead_items = []
                    if left_absorbable:
                        for score, extra_consumed in extend_absorb_path(
                            mpo_left, 1, 0, absorb_lookahead_depth - 1
                        ):
                            lookahead_items.append(
                                ("L", score, 1 + extra_consumed, score_absorb_candidate(mpo_left, absorb_score))
                            )
                    if right_absorbable:
                        for score, extra_consumed in extend_absorb_path(
                            mpo_right, 0, 1, absorb_lookahead_depth - 1
                        ):
                            lookahead_items.append(
                                ("R", score, 1 + extra_consumed, score_absorb_candidate(mpo_right, absorb_score))
                            )
                    lookahead_time_s = time.perf_counter() - lookahead_started
                    timing_totals["absorb_lookahead_time_s"] += lookahead_time_s
                    lookahead_candidates = len(lookahead_items)
                    if lookahead_items:
                        side_chosen, lookahead_score, lookahead_depth_used, _ = min(
                            lookahead_items,
                            key=lambda item: (item[1], -item[2], item[3]),
                        )
                        lookahead_choice = side_chosen
                    else:
                        side_chosen = min(eligible, key=lambda item: item[1])[0]
                else:
                    side_chosen = min(eligible, key=lambda item: item[1])[0]
                do_left = side_chosen == "L"
            else:
                do_left = counts_left < counts_right
        else:
            if mpo_left is None:
                do_left = False
            elif mpo_right is None:
                do_left = True
            elif (ii_right + ii_left) % flip_freq == 0:
                do_left = not do_left

        # Select the smallest one
        selected_counts = [counts_right, counts_left][int(do_left)]
        selected_max_bond = [max_bond_right, max_bond_left][int(do_left)]
        selected_absorbable = (
            selected_counts < unswap_threshold
            and (
                unswap_trigger_max_bond is None
                or selected_max_bond <= unswap_trigger_max_bond
            )
        )
        if selected_absorbable:
            if do_left:
                mpo_core = mpo_left
                # Update counts
                new_ops = dict(layers_left[ii_left].count_ops())
                new_us = count_work_ops(layers_left[ii_left])
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_left += new_us

                # Log
                side_chosen = "L"
                ii_left += 1
            else:
                mpo_core = mpo_right
                # Update counts
                new_ops = dict(layers_right[ii_right].count_ops())
                new_us = count_work_ops(layers_right[ii_right])
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_right += new_us
            
                # Log
                side_chosen = "R"
                ii_right += 1            
            
            logging.info((f"[{ii_right}R/{len(layers_right)}]" if side_chosen == "R" else f"[{ii_left}L/{len(layers_left)}]") +
                         f"(swap: {new_swaps}, u: {new_us} | c_u: {current_u_consumed} | t_u_l: {total_u_consumed_left}/{T_UL} | t_u_r: {total_u_consumed_right}/{T_UR} | t_u: {total_u_consumed}/{T_U}) -> " +
                         str(get_tn_info(mpo_core)))
            row = {"time": time.perf_counter() - t0, "stage": "absorbing", "absorb_side": side_chosen,
                                "it_left": ii_left, "it_right": ii_right, "layers_left": len(layers_left), "layers_right": len(layers_right),
                                "u_consumed_total_left": total_u_consumed_left, "u_consumed_total_right": total_u_consumed_right, "u_consumed_total": total_u_consumed,
                                "swap_consumed": new_swaps, "u_consumed": new_us, "u_consumed_after_unswap": current_u_consumed,
                                "probe_left_time_s": probe_left_time_s, "probe_right_time_s": probe_right_time_s,
                                "probe_total_time_s": probe_left_time_s + probe_right_time_s,
                                "probe_wall_time_s": probe_wall_time_s,
                                "probe_left_total_elems": counts_left if mpo_left is not None else None,
                                "probe_right_total_elems": counts_right if mpo_right is not None else None,
                                "probe_left_max_bond": max_bond_left if mpo_left is not None else None,
                                "probe_right_max_bond": max_bond_right if mpo_right is not None else None,
                                "probe_left_hit_max_bond": bool(max_bond is not None and mpo_left is not None and max_bond_left >= max_bond),
                                "probe_right_hit_max_bond": bool(max_bond is not None and mpo_right is not None and max_bond_right >= max_bond),
                                "probe_left_over_unswap_threshold": bool(mpo_left is not None and counts_left >= unswap_threshold),
                                "probe_right_over_unswap_threshold": bool(mpo_right is not None and counts_right >= unswap_threshold),
                                "selected_total_elems": selected_counts,
                                "selected_max_bond": selected_max_bond,
                                "selected_hit_max_bond": bool(max_bond is not None and selected_max_bond >= max_bond),
                                "compression_max_bond_limit": max_bond,
                                "compression_cutoff": cutoff,
                                "unswap_trigger_max_bond": unswap_trigger_max_bond,
                                "cycle_start_total_elems": cycle_start_total_elems,
                                "no_progress_unswap_cycles": no_progress_unswap_cycles,
                                "absorb_score_mode": absorb_score,
                                "probe_left_score": score_absorb_candidate(mpo_left, absorb_score) if mpo_left is not None else None,
                                "probe_right_score": score_absorb_candidate(mpo_right, absorb_score) if mpo_right is not None else None,
                                "absorb_lookahead_depth": absorb_lookahead_depth,
                                "absorb_lookahead_depth_used": lookahead_depth_used,
                                "absorb_lookahead_candidates": lookahead_candidates,
                                "absorb_lookahead_choice": lookahead_choice,
                                "absorb_lookahead_score": lookahead_score,
                                "absorb_lookahead_time_s": lookahead_time_s,
                                **get_tn_info(mpo_core)}
            stats_data.append(row)
            if on_stats is not None:
                on_stats(row)

            if max_work_gates is not None and total_u_consumed >= max_work_gates:
                termination_reason = "max_work_gates"
                termination_detail = (
                    f"consumed {total_u_consumed} work gates; "
                    f"target was {max_work_gates}"
                )
                break
        
        # Unswap if both sides go over the size budget
        else: 
            # Apply unswapping
            try:
                unswap_started = time.perf_counter()
                cycle_unswap_select_mode = unswap_select_mode
                if (
                    unswap_select_mode == "bond_route_proxy"
                    and unswap_route_proxy_max_cycles is not None
                    and unswap_cycles >= unswap_route_proxy_max_cycles
                ):
                    cycle_unswap_select_mode = "bond"
                mpo_core, (new_perm_left, new_perm_right), new_unswap_stats = unswap(
                    mpo_core,
                    hows=hows,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    max_its=max_its,
                    equal=equal,
                    to_backend=to_backend,
                    t0=t0,
                    stop_total_elems=unswap_stop_total_elems,
                    stop_max_bond=unswap_stop_max_bond,
                    probe_max_bond=unswap_probe_max_bond,
                    probe_cutoff=unswap_probe_cutoff,
                    adaptive_stop_max_bond=unswap_adaptive_stop_max_bond,
                    adaptive_stop_min_rel_improvement=unswap_adaptive_stop_min_rel_improvement,
                    adaptive_stop_min_iteration=unswap_adaptive_stop_min_iteration,
                    swap_apply_method=swap_apply_method,
                    swap_gate_representation=unswap_swap_representation,
                    unswap_select_mode=cycle_unswap_select_mode,
                    reuse_full_swap_probe=reuse_full_swap_probe,
                    unswap_hot_bonds=unswap_hot_bonds,
                    unswap_hot_radius=unswap_hot_radius,
                    unswap_pair_lookahead_limit=unswap_pair_lookahead_limit,
                    future_layers_left=layers_left[(ii_left):] + init_meas,
                    future_layers_right=layers_right[(ii_right):] + final_meas,
                    unswap_route_proxy_weight=unswap_route_proxy_weight,
                    unswap_route_proxy_lookahead=unswap_route_proxy_lookahead,
                    unswap_route_proxy_include_swaps=unswap_route_proxy_include_swaps,
                    unswap_route_proxy_allow_nonbond=unswap_route_proxy_allow_nonbond,
                    unswap_route_proxy_policy=unswap_route_proxy_policy,
                    unswap_route_proxy_min_benefit=unswap_route_proxy_min_benefit,
                    unswap_route_proxy_max_bond_loss=unswap_route_proxy_max_bond_loss,
                    unswap_route_proxy_protect_gain=unswap_route_proxy_protect_gain,
                    unswap_alignment_weight=unswap_alignment_weight,
                    unswap_alignment_protect_gain=unswap_alignment_protect_gain,
                    unswap_alignment_max_replacements=unswap_alignment_max_replacements,
                    unswap_alignment_tie_loss=unswap_alignment_tie_loss,
                )
                unswap_time_s = time.perf_counter() - unswap_started
                timing_totals["unswap_time_s"] += unswap_time_s
                unswap_cycles += 1
                stats_data += new_unswap_stats
                if on_stats is not None:
                    for row in new_unswap_stats:
                        on_stats(row)
                row = {
                    "time": time.perf_counter() - t0,
                    "stage": "unswap_cycle_summary",
                    "unswap_cycle": unswap_cycles,
                    "unswap_cycle_time_s": unswap_time_s,
                    "u_consumed_total_left": total_u_consumed_left,
                    "u_consumed_total_right": total_u_consumed_right,
                    "u_consumed_total": total_u_consumed,
                    "gates_consumed": total_u_consumed,
                    "total_work_gates": T_U,
                    "remaining_work_gates": T_U - total_u_consumed,
                    "cycle_gates_consumed": current_u_consumed,
                    "compression_max_bond_limit": max_bond,
                    "compression_cutoff": cutoff,
                    "hit_max_bond": bool(max_bond is not None and mpo_core.max_bond() >= max_bond),
                    **get_tn_info(mpo_core),
                }
                stats_data.append(row)
                if on_stats is not None:
                    on_stats(row)
            except KeyboardInterrupt:
                termination_reason = "keyboard_interrupt"
                termination_detail = "interrupted during unswap"
                break        
            def rewire_remaining(side):
                if side == "left":
                    remaining_layers = layers_left[(ii_left):] + init_meas
                    perm = new_perm_left
                else:
                    remaining_layers = layers_right[(ii_right):] + final_meas
                    perm = new_perm_right

                cycle_route_candidates = route_candidates
                cycle_route_score = route_score
                cycle_route_seed = post_rewire_seed

                logging.info(
                    "[start rewire](phase=post_unswap, cycle=%s, side=%s, layers=%s, "
                    "trials=%s, seed=%s, parallel=%s, route_candidates=%s, route_score=%s)",
                    unswap_cycles,
                    side,
                    len(remaining_layers),
                    post_rewire_sabre_trials,
                    cycle_route_seed,
                    adaptive_parallel_rewire_active,
                    cycle_route_candidates,
                    cycle_route_score,
                )
                rewire_started = time.perf_counter()
                new_layers, route_metadata = rewire_layers_scored(
                    remaining_layers,
                    perm,
                    seed=cycle_route_seed,
                    sabre_trials=post_rewire_sabre_trials,
                    sabre_heuristic=sabre_heuristic,
                    route_candidates=cycle_route_candidates,
                    route_seed_stride=route_seed_stride,
                    route_score=cycle_route_score,
                    route_score_lookahead=route_score_lookahead,
                    mpo_core=mpo_core,
                    side=side,
                    q2c=q2c,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    parallel_route_candidates=parallel_route_candidates,
                    route_candidate_workers=route_candidate_workers,
                )
                rewire_time_s = time.perf_counter() - rewire_started
                logging.info(
                    "[end rewire](phase=post_unswap, cycle=%s, side=%s, routed_layers=%s, elapsed_s=%.2f)",
                    unswap_cycles,
                    side,
                    len(new_layers),
                    rewire_time_s,
                )
                return new_layers, rewire_time_s, {
                    "time": time.perf_counter() - t0,
                    "stage": "rewiring",
                    "rewire_side": side,
                    "rewire_phase": "post_unswap",
                    "rewire_time_s": rewire_time_s,
                    "sabre_trials": post_rewire_sabre_trials,
                    "unswap_cycle": unswap_cycles,
                    "remaining_layers_before_rewire": len(remaining_layers),
                    "parallel_rewire": adaptive_parallel_rewire_active,
                    "adaptive_parallel_rewire": adaptive_parallel_rewire,
                    "adaptive_parallel_rewire_probed": adaptive_parallel_rewire_probed,
                    "cycle_work_consumed_before_rewire": current_u_consumed,
                    **route_metadata,
                }

            can_parallel_rewire = (
                adaptive_parallel_rewire_active
                and ii_left < len(layers_left)
                and ii_right < len(layers_right)
                and route_candidates <= 1
                and route_score == "none"
            )
            rewire_wall_started = time.perf_counter()
            if can_parallel_rewire:
                left_future = rewire_executor.submit(rewire_remaining, "left")
                right_future = rewire_executor.submit(rewire_remaining, "right")
                rewire_results = {
                    "left": left_future.result(),
                    "right": right_future.result(),
                }
            else:
                rewire_results = {}
                if ii_left < len(layers_left):
                    rewire_results["left"] = rewire_remaining("left")
                if ii_right < len(layers_right):
                    rewire_results["right"] = rewire_remaining("right")
            rewire_wall_time_s = time.perf_counter() - rewire_wall_started
            timing_totals["post_unswap_rewire_wall_time_s"] += rewire_wall_time_s

            if "left" in rewire_results:
                layers_left, rewire_time_s, row = rewire_results["left"]
                row["rewire_wall_time_s"] = rewire_wall_time_s
                timing_totals["post_unswap_rewire_time_s"] += rewire_time_s
                stats_data.append(row)
                if on_stats is not None:
                    on_stats(row)
                init_meas = layers_left[-2:]
                layers_left = layers_left[:-2]
            else:
                layers_left = []

            if "right" in rewire_results:
                layers_right, rewire_time_s, row = rewire_results["right"]
                row["rewire_wall_time_s"] = rewire_wall_time_s
                timing_totals["post_unswap_rewire_time_s"] += rewire_time_s
                stats_data.append(row)
                if on_stats is not None:
                    on_stats(row)
                final_meas = layers_right[-2:]
                layers_right = layers_right[:-2]
            else:
                layers_right = []

            cycle_work_consumed = current_u_consumed
            row = {
                "time": time.perf_counter() - t0,
                "stage": "cycle_progress",
                "unswap_cycle": unswap_cycles,
                "u_consumed_total_left": total_u_consumed_left,
                "u_consumed_total_right": total_u_consumed_right,
                "u_consumed_total": total_u_consumed,
                "gates_consumed": total_u_consumed,
                "total_work_gates": T_U,
                "remaining_work_gates": T_U - total_u_consumed,
                "cycle_gates_consumed": cycle_work_consumed,
                "no_progress_unswap_cycles": no_progress_unswap_cycles,
                "cycle_start_total_elems": cycle_start_total_elems,
                "rewire_wall_time_s": rewire_wall_time_s,
                "compression_max_bond_limit": max_bond,
                "compression_cutoff": cutoff,
                "hit_max_bond": bool(max_bond is not None and mpo_core.max_bond() >= max_bond),
                **get_tn_info(mpo_core),
            }
            stats_data.append(row)
            if on_stats is not None:
                on_stats(row)
            
            ii_left = 0
            ii_right = 0
            if cycle_work_consumed == 0:
                no_progress_unswap_cycles += 1
            else:
                no_progress_unswap_cycles = 0

            if (
                abort_after_no_progress_unswap_cycles is not None
                and abort_after_no_progress_unswap_cycles >= 0
                and no_progress_unswap_cycles >= abort_after_no_progress_unswap_cycles
            ):
                termination_reason = "no_progress_cycle_limit"
                # Classify the stall. Two very different failure modes share
                # this termination reason, and they need opposite remedies:
                #
                #   (A) entanglement blow-up: the MPO is pinned near max_bond
                #       and/or the unswap_threshold, so every candidate absorb
                #       overflows and no block can be consumed. This is the
                #       cutoff-sensitive case -- a different cutoff lands on a
                #       cheaper truncation trajectory.
                #
                #   (B) swap-layer thrash: the MPO is small (well below both
                #       the bond cap and the element threshold) but consecutive
                #       cycles absorbed only routing SWAP layers before reaching
                #       the next consolidated 2q block, so zero *work* gates
                #       were consumed. Changing the cutoff does nothing here --
                #       the fix is to let the loop run more cycles via
                #       --abort-after-no-progress-unswap-cycles (or disable it
                #       with a negative value).
                core_elems = elem_counts(mpo_core)
                core_max_bond = mpo_core.max_bond()
                near_bond_cap = max_bond is not None and core_max_bond >= 0.5 * max_bond
                near_threshold = core_elems >= 0.5 * unswap_threshold
                if near_bond_cap or near_threshold:
                    stall_mode = "entanglement_blowup"
                    remedy = (
                        f"The MPO is pinned near the size limits "
                        f"(max_bond={core_max_bond} vs cap {max_bond}, "
                        f"elems={core_elems} vs unswap_threshold "
                        f"{unswap_threshold:.0f}), so no block can be absorbed "
                        f"without overflowing. This is the cutoff-sensitive "
                        f"branch: retry with a different --cutoff (verified "
                        f"Apple Silicon Accelerate values are in BENCHMARKS.md; "
                        f"start with --cutoff 0.0006), or raise --max-bond / "
                        f"--unswap-threshold."
                    )
                else:
                    stall_mode = "swap_thrash"
                    remedy = (
                        f"The MPO is small at the stall "
                        f"(max_bond={core_max_bond} vs cap {max_bond}, "
                        f"elems={core_elems} vs unswap_threshold "
                        f"{unswap_threshold:.0f}) -- the last cycles absorbed "
                        f"only routing SWAP layers before reaching the next 2q "
                        f"block, so zero work gates were consumed. Changing "
                        f"--cutoff will NOT help. Raise "
                        f"--abort-after-no-progress-unswap-cycles (e.g. 20) or "
                        f"set it negative to disable the guard and let the "
                        f"reroute push past this patch."
                    )
                termination_detail = (
                    f"{no_progress_unswap_cycles} consecutive unswap cycles "
                    f"consumed zero work gates at cutoff={cutoff} "
                    f"[mode={stall_mode}]. {remedy}"
                )
                logging.error(
                    "aborting after %s consecutive no-progress unswap cycles "
                    "(limit=%s, consumed=%s/%s, cutoff=%s, mode=%s). %s",
                    no_progress_unswap_cycles,
                    abort_after_no_progress_unswap_cycles,
                    total_u_consumed,
                    T_U,
                    cutoff,
                    stall_mode,
                    remedy,
                )
                row = {
                    "time": time.perf_counter() - t0,
                    "stage": "termination",
                    "termination_reason": termination_reason,
                    "termination_detail": termination_detail,
                    "stall_mode": stall_mode,
                    "unswap_cycle": unswap_cycles,
                    "u_consumed_total_left": total_u_consumed_left,
                    "u_consumed_total_right": total_u_consumed_right,
                    "u_consumed_total": total_u_consumed,
                    "gates_consumed": total_u_consumed,
                    "total_work_gates": T_U,
                    "remaining_work_gates": T_U - total_u_consumed,
                    "no_progress_unswap_cycles": no_progress_unswap_cycles,
                    "abort_after_no_progress_unswap_cycles": abort_after_no_progress_unswap_cycles,
                    "compression_max_bond_limit": max_bond,
                    "compression_cutoff": cutoff,
                    "hit_max_bond": bool(max_bond is not None and mpo_core.max_bond() >= max_bond),
                    **get_tn_info(mpo_core),
                }
                stats_data.append(row)
                if on_stats is not None:
                    on_stats(row)
                break
            current_u_consumed = 0
            cycle_start_total_elems = elem_counts(mpo_core)

            # Stop early if there are few gates left
            if (T_U - total_u_consumed) <= early_stopping_gates:
                termination_reason = "early_stopping_gates"
                termination_detail = (
                    f"{T_U - total_u_consumed} work gates remain; "
                    f"early_stopping_gates={early_stopping_gates}"
                )
                break

            if max_unswap_cycles is not None and unswap_cycles >= max_unswap_cycles:
                termination_reason = "max_unswap_cycles"
                termination_detail = (
                    f"completed {unswap_cycles} unswap cycles; "
                    f"limit was {max_unswap_cycles}"
                )
                break

            if max_work_gates is not None and total_u_consumed >= max_work_gates:
                termination_reason = "max_work_gates"
                termination_detail = (
                    f"consumed {total_u_consumed} work gates; "
                    f"target was {max_work_gates}"
                )
                break
    
    # Remove any leftover layers
    layers_left = layers_left[(ii_left):] if ii_left < len(layers_left) else []
    layers_left += init_meas
    layers_right = layers_right[(ii_right):] if ii_right < len(layers_right) else []
    layers_right += final_meas

    logging.info(f"[end compressing](left: {len(layers_left)}, right: {len(layers_right)}) -> " + str(get_tn_info(mpo_core)))
    timing_totals["rewire_time_s"] = timing_totals["initial_rewire_time_s"] + timing_totals["post_unswap_rewire_time_s"]
    timing_totals["accounted_time_s"] = timing_totals["rewire_time_s"] + timing_totals["absorb_probe_wall_time_s"] + timing_totals["unswap_time_s"]
    timing_totals["rewire_wall_time_s"] = timing_totals["initial_rewire_wall_time_s"] + timing_totals["post_unswap_rewire_wall_time_s"]
    timing_totals["accounted_wall_time_s"] = timing_totals["rewire_wall_time_s"] + timing_totals["absorb_probe_wall_time_s"] + timing_totals["unswap_time_s"]
    if probe_executor is not None:
        probe_executor.shutdown()
    if rewire_executor is not None:
        rewire_executor.shutdown()
    if not any(row.get("stage") == "termination" for row in stats_data):
        stats_data.append({
            "time": time.perf_counter() - t0,
            "stage": "termination",
            "termination_reason": termination_reason,
            "termination_detail": termination_detail,
            "unswap_cycle": unswap_cycles,
            "u_consumed_total_left": total_u_consumed_left,
            "u_consumed_total_right": total_u_consumed_right,
            "u_consumed_total": total_u_consumed,
            "gates_consumed": total_u_consumed,
            "total_work_gates": T_U,
            "remaining_work_gates": T_U - total_u_consumed,
            "no_progress_unswap_cycles": no_progress_unswap_cycles,
            "abort_after_no_progress_unswap_cycles": abort_after_no_progress_unswap_cycles,
            "compression_max_bond_limit": max_bond,
            "compression_cutoff": cutoff,
            "hit_max_bond": bool(max_bond is not None and mpo_core.max_bond() >= max_bond),
            **get_tn_info(mpo_core),
        })

    stats_data.append({
        "time": time.perf_counter() - t0,
        "stage": "timing_summary",
        **timing_totals,
    })

    return mpo_core, layers_left, layers_right, stats_data


def mpo_to_mps(mpo_core, layers_left, layers_right, max_bond=4096, cutoff=0.001, to_backend=None):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    # Use the compressed MPO to get the MPS by applying it to |0> state
    final_mps = quimb_circuit(
        QuantumCircuit(len(mpo_core.sites)),
        quimb_circuit_class=CircuitMPS,
        to_backend=to_backend,
    ).psi

    # First take the leftover front layers
    layers_left = list(iter_layers(merge_layers(layers_left).inverse())) if len(layers_left) > 0 else []
    
    for ii_left in range(len(layers_left)):
        l_left = layers_left[ii_left]
        new_ops = dict(l_left.count_ops())
        layer_mpo = mpo_from_circuit(q2c(l_left))
        final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
        logging.info(f"[Left {ii_left} / {len(layers_left)}] -> " + str(get_tn_info(final_mps)))

    logging.info("[Left MPS] -> " + str(get_tn_info(final_mps)))

    # Then apply the compressed MPO to the layers
    final_mps = mpo_core.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
    logging.info("[Left MPS + Core MPO] -> " + str(get_tn_info(final_mps)))

    # Then iterate through final layers if there are any
    final_meas = []
    for ii_right in range(len(layers_right)):
        l_right = layers_right[ii_right]
        new_ops = dict(l_right.count_ops())
        if "barrier" in new_ops or "measure" in new_ops:
            final_meas.append(l_right)
        else:
            layer_mpo = mpo_from_circuit(q2c(l_right))
            final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
            logging.info(f"[Front MPS + Core MPO + Right {ii_right} / {len(layers_right)}] -> " + str(get_tn_info(final_mps)))
    
    logging.info(f"[Front MPS + Core MPO + Right MPS] -> " + str(get_tn_info(final_mps)))

    # Extract final permutation from measurements
    final_perm = [g.qubits[0]._index for g in final_meas[-1]]

    # Return MPS and final perm
    return final_mps, final_perm

