#!/bin/bash
#SBATCH --job-name=unfold-cpu-symbolic
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cpu_symbolic_%j.out
#SBATCH --error=logs/cpu_symbolic_%j.err

set -euo pipefail

cd /home/dallin/Documents/Classes/Graph_Theory/isospectral_pst/unfolding/unfolding_ml
source .venv/bin/activate

mkdir -p logs results

python benchmarks/cpu_symbolic_benchmark.py \
  --n-instances 10 \
  --n 8 \
  --p 0.5 \
  --seed 1234 \
  --kept-vertex 0 \
  --n-atlas-max 7 \
  --max-checks 2000000 \
  --time-limit-sec 1800 \
  --output-dir results
