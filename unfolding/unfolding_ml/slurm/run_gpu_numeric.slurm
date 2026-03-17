#!/bin/bash
#SBATCH --job-name=unfold-gpu-numeric
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu_numeric_%j.out
#SBATCH --error=logs/gpu_numeric_%j.err

set -euo pipefail

cd /home/dallin/Documents/Classes/Graph_Theory/isospectral_pst/unfolding/unfolding_ml
source .venv/bin/activate

mkdir -p logs results

python benchmarks/gpu_numeric_benchmark.py \
  --n-instances 20 \
  --n 10 \
  --p 0.5 \
  --seed 1234 \
  --kept-vertex 0 \
  --n-candidates 200000 \
  --candidate-p 0.5 \
  --tol 1e-8 \
  --backend auto \
  --output-dir results
