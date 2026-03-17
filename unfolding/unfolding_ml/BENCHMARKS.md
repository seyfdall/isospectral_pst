# Unfolding Benchmark Tracks (CPU Symbolic vs GPU Numeric)

This workspace now includes two benchmark tracks for side-by-side supercomputer runs:

- **CPU Symbolic (`benchmarks/cpu_symbolic_benchmark.py`)**
  - Exact symbolic reduction/equality checks with SymPy.
  - Enumerates `(H, c)` using NetworkX graph atlas up to `n_atlas_max <= 7`.
  - Uses guardrails: `--max-checks`, `--time-limit-sec`.

- **GPU Numeric (`benchmarks/gpu_numeric_benchmark.py`)**
  - Numeric reduction matching over sampled `lambda` values.
  - Batch candidate scoring on GPU via JAX when available; NumPy fallback on CPU.
  - Uses tolerance-based matching (`--tol`) over many random candidate graphs.

Both write compatible outputs into `results/`:

- `*_records_<timestamp>.jsonl` per-instance rows
- `*_summary_<timestamp>.json` aggregate metrics
- `*_instances_<timestamp>.json` dataset instances used

## Local Runs

From workspace root:

```bash
source .venv/bin/activate

python benchmarks/cpu_symbolic_benchmark.py \
  --n-instances 3 --n 5 --p 0.5 --seed 1234 \
  --kept-vertex 0 --n-atlas-max 5 \
  --max-checks 200000 --time-limit-sec 300 \
  --output-dir results

python benchmarks/gpu_numeric_benchmark.py \
  --n-instances 8 --n 10 --p 0.5 --seed 1234 \
  --kept-vertex 0 --n-candidates 50000 \
  --candidate-p 0.5 --tol 1e-8 --backend auto \
  --output-dir results
```

## Compare Summaries

After both runs complete, compare their summary files:

```bash
python benchmarks/compare_benchmarks.py \
  --cpu-summary results/cpu_symbolic_summary_<timestamp>.json \
  --gpu-summary results/gpu_numeric_summary_<timestamp>.json \
  --output-json results/benchmark_comparison.json
```

## SLURM Runs

Template jobs:

- `slurm/run_cpu_symbolic.slurm`
- `slurm/run_gpu_numeric.slurm`

Submit with:

```bash
sbatch slurm/run_cpu_symbolic.slurm
sbatch slurm/run_gpu_numeric.slurm
```

## Notes

- CPU symbolic is exact but limited by atlas size and symbolic complexity.
- GPU numeric is scalable but approximate (numeric tolerance), not symbolic proof.
- For fair comparison, keep graph generation parameters aligned (`--n`, `--p`, `--seed`, `--kept-vertex`).
