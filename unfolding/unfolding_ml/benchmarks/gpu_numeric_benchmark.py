from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np

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


try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except Exception:
    HAS_JAX = False
    jax = None
    jnp = None


def build_random_candidates(
    *,
    n: int,
    n_candidates: int,
    p: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = np.zeros((n_candidates, n, n), dtype=np.float32)

    tri_rows, tri_cols = np.triu_indices(n, k=1)
    n_edges = len(tri_rows)
    edge_bits = rng.binomial(1, p, size=(n_candidates, n_edges)).astype(np.float32)

    for i in range(n_candidates):
        A = np.zeros((n, n), dtype=np.float32)
        A[tri_rows, tri_cols] = edge_bits[i]
        A[tri_cols, tri_rows] = edge_bits[i]
        candidates[i] = A

    return candidates


def reduction_values_numpy(
    A_batch: np.ndarray,
    kept_idx: int,
    lambdas: np.ndarray,
    regularization_eps: float,
) -> np.ndarray:
    batch, n, _ = A_batch.shape
    sbar = [i for i in range(n) if i != kept_idx]

    c = A_batch[:, sbar, kept_idx]
    A_H = A_batch[:, sbar, :][:, :, sbar]

    out = np.zeros((batch, len(lambdas)), dtype=np.float64)
    eye = np.eye(len(sbar), dtype=np.float64)

    for li, lam in enumerate(lambdas):
        M = A_H.astype(np.float64) - lam * eye[None, :, :] + regularization_eps * eye[None, :, :]
        x = np.linalg.solve(M, c.astype(np.float64)[..., None])[..., 0]
        vals = -np.sum(c.astype(np.float64) * x, axis=1)
        out[:, li] = vals

    return out


def build_jax_reduction_fn(n: int, kept_idx: int, regularization_eps: float):
    sbar = [i for i in range(n) if i != kept_idx]
    sbar_jnp = jnp.array(sbar)

    def per_graph(A: jnp.ndarray, lambdas: jnp.ndarray) -> jnp.ndarray:
        c = A[sbar_jnp, kept_idx]
        A_H = A[sbar_jnp[:, None], sbar_jnp[None, :]]

        def one_lambda(lam):
            M = (
                A_H
                - lam * jnp.eye(A_H.shape[0], dtype=A_H.dtype)
                + regularization_eps * jnp.eye(A_H.shape[0], dtype=A_H.dtype)
            )
            x = jnp.linalg.solve(M, c)
            return -jnp.dot(c, x)

        return jax.vmap(one_lambda)(lambdas)

    return jax.vmap(per_graph, in_axes=(0, None))


def run(args: argparse.Namespace) -> None:
    instances = generate_connected_er_instances(
        n_instances=args.n_instances,
        n=args.n,
        p=args.p,
        seed=args.seed,
        kept_vertex=args.kept_vertex,
    )

    lambdas = np.array(args.lambdas, dtype=np.float64)
    records: list[BenchmarkRecord] = []

    use_jax = HAS_JAX and args.backend in {"jax", "auto"}
    backend_name = "jax" if use_jax else "numpy"

    if use_jax:
        print(f"JAX backend enabled. Device(s): {jax.devices()}")
        batched_fn = build_jax_reduction_fn(args.n, args.kept_vertex, args.regularization_eps)
    else:
        print("Using NumPy backend (CPU fallback).")

    for inst in instances:
        t0 = time.time()
        status = "ok"
        n_candidates_evaluated = 0
        n_matches = 0
        recovered_original = False
        extra: dict[str, object] = {}

        try:
            G_source = graph_from_instance(inst)
            A_source = nx.to_numpy_array(
                G_source, nodelist=range(inst.n), dtype=float
            ).astype(np.float32)

            candidate_batch = build_random_candidates(
                n=inst.n,
                n_candidates=args.n_candidates,
                p=args.candidate_p,
                seed=inst.seed + args.candidate_seed_offset,
            )

            candidate_batch[0] = A_source

            if use_jax:
                target_vals = np.array(
                    batched_fn(jnp.asarray(A_source[None, :, :]), jnp.asarray(lambdas))[0]
                )
                cand_vals = np.array(
                    batched_fn(jnp.asarray(candidate_batch), jnp.asarray(lambdas))
                )
            else:
                target_vals = reduction_values_numpy(
                    A_source[None, :, :], args.kept_vertex, lambdas, args.regularization_eps
                )[0]
                cand_vals = reduction_values_numpy(
                    candidate_batch, args.kept_vertex, lambdas, args.regularization_eps
                )

            residuals = np.max(np.abs(cand_vals - target_vals[None, :]), axis=1)
            match_mask = residuals <= args.tol

            n_candidates_evaluated = candidate_batch.shape[0]
            n_matches = int(np.sum(match_mask))
            recovered_original = bool(match_mask[0])

            extra = {
                "backend": backend_name,
                "lambdas": list(map(float, lambdas)),
                "tol": args.tol,
                "regularization_eps": args.regularization_eps,
                "best_residual": float(np.min(residuals)),
                "median_residual": float(np.median(residuals)),
            }

        except Exception as exc:
            status = "error"
            extra = {"error": str(exc), "backend": backend_name}

        elapsed = time.time() - t0
        records.append(
            BenchmarkRecord(
                track="gpu_numeric",
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
            f"[gpu_numeric] instance={inst.instance_id} status={status} "
            f"elapsed={elapsed:.2f}s candidates={n_candidates_evaluated} "
            f"matches={n_matches} recovered_original={recovered_original}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso().replace(":", "-")

    rows = records_to_rows(records)
    summary = summarize(records)
    metadata = {
        "track": "gpu_numeric",
        "timestamp": now_iso(),
        "args": vars(args),
        "summary": summary,
        "backend": backend_name,
    }

    write_jsonl(output_dir / f"gpu_numeric_records_{timestamp}.jsonl", rows)
    write_json(output_dir / f"gpu_numeric_summary_{timestamp}.json", metadata)
    write_json(
        output_dir / f"gpu_numeric_instances_{timestamp}.json",
        {"instances": [asdict(i) for i in instances]},
    )

    print("\nGPU-first numeric benchmark complete")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-first numeric unfolding benchmark")
    parser.add_argument("--n-instances", type=int, default=8)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kept-vertex", type=int, default=0)

    parser.add_argument("--n-candidates", type=int, default=50000)
    parser.add_argument("--candidate-p", type=float, default=0.5)
    parser.add_argument("--candidate-seed-offset", type=int, default=10_000)

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[-5.0, -3.0, -1.5, 1.25, 2.5, 4.0],
    )
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--regularization-eps", type=float, default=1e-8)
    parser.add_argument("--backend", choices=["auto", "jax", "numpy"], default="auto")

    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
