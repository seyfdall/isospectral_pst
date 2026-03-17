from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import itertools
import networkx as nx
import numpy as np
import sympy as sp

from common import (
    BenchmarkRecord,
    generate_connected_er_instances,
    graph_from_instance,
    now_iso,
    records_to_rows,
    summarize,
    write_json,
    write_jsonl,
)


def compute_reduction(G: nx.Graph, kept_vertex: int, lam: sp.Symbol) -> sp.Expr:
    nodes = list(G.nodes())
    s = kept_vertex
    sbar = [v for v in nodes if v != s]

    A = sp.Matrix(nx.to_numpy_array(G, nodelist=nodes, dtype=int).tolist())
    idx = {v: i for i, v in enumerate(nodes)}
    s_idx = idx[s]
    sbar_idx = [idx[v] for v in sbar]

    a_ss = A[s_idx, s_idx]
    c = sp.Matrix([A[i, s_idx] for i in sbar_idx])
    A_H = sp.Matrix([[A[i, j] for j in sbar_idx] for i in sbar_idx])

    M = A_H - lam * sp.eye(len(sbar))
    return sp.cancel(sp.together(a_ss - (c.T * M.inv() * c)[0, 0]))


def assemble_unfolding(H: nx.Graph, c_tuple: tuple[int, ...]) -> np.ndarray:
    k = H.number_of_nodes()
    A = np.zeros((k + 1, k + 1), dtype=int)
    for i, ci in enumerate(c_tuple):
        A[0, i + 1] = ci
        A[i + 1, 0] = ci
    A_H = nx.to_numpy_array(H, nodelist=range(k), dtype=int)
    A[1:, 1:] = A_H
    return A


def enumerate_unfoldings(
    target_r: sp.Expr,
    lam: sp.Symbol,
    n_atlas_max: int,
    max_checks: int,
    time_limit_sec: float,
) -> tuple[list[np.ndarray], int, bool]:
    target = sp.cancel(sp.together(target_r))
    start = time.time()
    checks = 0
    timed_out = False

    matches: list[np.ndarray] = []

    for G0 in nx.graph_atlas_g():
        k = G0.number_of_nodes()
        if k == 0:
            continue
        if k > n_atlas_max:
            break

        G = nx.convert_node_labels_to_integers(G0)
        if nx.number_of_selfloops(G) > 0:
            continue

        A = sp.Matrix(nx.to_numpy_array(G, nodelist=range(k), dtype=int).tolist())
        M = A - lam * sp.eye(k)

        try:
            M_inv = M.inv()
        except Exception:
            continue

        for c_tuple in itertools.product([0, 1], repeat=k):
            if all(x == 0 for x in c_tuple):
                continue

            checks += 1
            if checks >= max_checks:
                timed_out = True
                return matches, checks, timed_out
            if time.time() - start >= time_limit_sec:
                timed_out = True
                return matches, checks, timed_out

            c = sp.Matrix(list(c_tuple))
            f = sp.cancel(sp.together(-(c.T * M_inv * c)[0, 0]))
            if sp.simplify(f - target) == 0:
                matches.append(assemble_unfolding(G, c_tuple))

    return matches, checks, timed_out


def deduplicate_graphs(A_list: list[np.ndarray]) -> list[np.ndarray]:
    reps: list[np.ndarray] = []
    rep_graphs: list[nx.Graph] = []

    for A in A_list:
        G = nx.from_numpy_array(A)
        is_new = True
        for rep_g in rep_graphs:
            if nx.is_isomorphic(G, rep_g):
                is_new = False
                break
        if is_new:
            reps.append(A)
            rep_graphs.append(G)

    return reps


def run(args: argparse.Namespace) -> None:
    lam = sp.symbols("lambda")
    instances = generate_connected_er_instances(
        n_instances=args.n_instances,
        n=args.n,
        p=args.p,
        seed=args.seed,
        kept_vertex=args.kept_vertex,
    )

    records: list[BenchmarkRecord] = []

    for inst in instances:
        t0 = time.time()
        status = "ok"
        n_candidates_evaluated = 0
        n_matches = 0
        recovered_original = False
        extra: dict[str, object] = {}

        try:
            G_source = graph_from_instance(inst)
            target_r = compute_reduction(G_source, inst.kept_vertex, lam)

            matches, checks, timed_out = enumerate_unfoldings(
                target_r=target_r,
                lam=lam,
                n_atlas_max=args.n_atlas_max,
                max_checks=args.max_checks,
                time_limit_sec=args.time_limit_sec,
            )
            unique_matches = deduplicate_graphs(matches)

            source_full = nx.convert_node_labels_to_integers(G_source)
            n_candidates_evaluated = checks
            n_matches = len(unique_matches)
            recovered_original = any(
                nx.is_isomorphic(nx.from_numpy_array(A), source_full) for A in unique_matches
            )

            extra = {
                "target_r": str(target_r),
                "timed_out": timed_out,
                "n_raw_matches": len(matches),
            }
            if timed_out:
                status = "timeout"

        except Exception as exc:
            status = "error"
            extra = {"error": str(exc)}

        elapsed = time.time() - t0
        records.append(
            BenchmarkRecord(
                track="cpu_symbolic",
                instance_id=inst.instance_id,
                n=inst.n,
                p=inst.p,
                seed=inst.seed,
                kept_vertex=inst.kept_vertex,
                elapsed_sec=elapsed,
                status=status,
                n_candidates_evaluated=n_candidates_evaluated,
                n_matches=n_matches,
                recovered_original=recovered_original,
                extra=extra,
            )
        )

        print(
            f"[cpu_symbolic] instance={inst.instance_id} status={status} "
            f"elapsed={elapsed:.2f}s checks={n_candidates_evaluated} matches={n_matches} "
            f"recovered_original={recovered_original}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso().replace(":", "-")

    rows = records_to_rows(records)
    summary = summarize(records)
    metadata = {
        "track": "cpu_symbolic",
        "timestamp": now_iso(),
        "args": vars(args),
        "summary": summary,
    }

    write_jsonl(output_dir / f"cpu_symbolic_records_{timestamp}.jsonl", rows)
    write_json(output_dir / f"cpu_symbolic_summary_{timestamp}.json", metadata)
    write_json(
        output_dir / f"cpu_symbolic_instances_{timestamp}.json",
        {"instances": [asdict(i) for i in instances]},
    )

    print("\nCPU symbolic benchmark complete")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU symbolic unfolding benchmark")
    parser.add_argument("--n-instances", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kept-vertex", type=int, default=0)

    parser.add_argument("--n-atlas-max", type=int, default=5)
    parser.add_argument("--max-checks", type=int, default=200000)
    parser.add_argument("--time-limit-sec", type=float, default=300.0)

    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
