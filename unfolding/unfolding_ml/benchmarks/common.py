from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class GraphInstance:
    instance_id: int
    n: int
    p: float
    seed: int
    kept_vertex: int
    edges: list[tuple[int, int]]


@dataclass
class BenchmarkRecord:
    track: str
    instance_id: int
    n: int
    p: float
    seed: int
    kept_vertex: int
    elapsed_sec: float
    status: str
    n_candidates_evaluated: int
    n_matches: int
    recovered_original: bool
    extra: dict[str, Any]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row)) + "\n")


def summarize(records: list[BenchmarkRecord]) -> dict[str, Any]:
    if not records:
        return {
            "n_instances": 0,
            "n_success": 0,
            "n_failed": 0,
            "mean_elapsed_sec": 0.0,
            "total_elapsed_sec": 0.0,
            "recovery_rate": 0.0,
        }

    total_elapsed = sum(r.elapsed_sec for r in records)
    n_success = sum(1 for r in records if r.status == "ok")
    n_failed = len(records) - n_success
    recovery_rate = sum(1 for r in records if r.recovered_original) / len(records)

    return {
        "n_instances": len(records),
        "n_success": n_success,
        "n_failed": n_failed,
        "mean_elapsed_sec": total_elapsed / len(records),
        "total_elapsed_sec": total_elapsed,
        "recovery_rate": recovery_rate,
    }


def records_to_rows(records: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]


def generate_connected_er_instances(
    *,
    n_instances: int,
    n: int,
    p: float,
    seed: int,
    kept_vertex: int = 0,
    max_tries_per_instance: int = 200,
) -> list[GraphInstance]:
    rng = random.Random(seed)
    instances: list[GraphInstance] = []
    next_seed = seed

    for instance_id in range(n_instances):
        graph = None
        for _ in range(max_tries_per_instance):
            candidate_seed = rng.randint(0, 10**9)
            graph_candidate = nx.erdos_renyi_graph(n, p, seed=candidate_seed)
            if nx.is_connected(graph_candidate):
                graph = graph_candidate
                next_seed = candidate_seed
                break

        if graph is None:
            raise RuntimeError(
                f"Failed to find connected ER graph for instance {instance_id} after "
                f"{max_tries_per_instance} attempts."
            )

        instances.append(
            GraphInstance(
                instance_id=instance_id,
                n=n,
                p=p,
                seed=next_seed,
                kept_vertex=kept_vertex,
                edges=sorted(tuple(sorted(e)) for e in graph.edges()),
            )
        )

    return instances


def graph_from_instance(instance: GraphInstance) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(instance.n))
    graph.add_edges_from(instance.edges)
    return graph
