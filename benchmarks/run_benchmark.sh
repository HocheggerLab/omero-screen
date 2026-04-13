#!/bin/bash
#SBATCH -J omero-benchmark
#SBATCH -o /mnt/lustre/users/gdsc/hh65/omero-screen/benchmarks/slurm_%j.out
#SBATCH -e /mnt/lustre/users/gdsc/hh65/omero-screen/benchmarks/slurm_%j.err
#SBATCH --mail-user=hh65@sussex.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# Usage:
#   sbatch benchmarks/run_benchmark.sh                         # defaults
#   sbatch benchmarks/run_benchmark.sh --plate-id 3802
#   sbatch benchmarks/run_benchmark.sh --plate-id 3802 --cellpose cp4
#   sbatch benchmarks/run_benchmark.sh --plate-id 3802 --no-gpu --repeats 5
#
# All extra arguments are forwarded to benchmark_repeats.py.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/mnt/lustre/users/gdsc/${USER}/omero-screen"
cd "${REPO_ROOT}"

# Load internet proxy so compute nodes can reach the OMERO server
module load proxy/public 2>/dev/null || true

# Activate the uv-managed virtual environment
source "${REPO_ROOT}/.venv/bin/activate"

echo "========================================"
echo "  OMERO-Screen Benchmark Job"
echo "  Host:    $(hostname)"
echo "  Node:    ${SLURMD_NODENAME:-unknown}"
echo "  GPUs:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Started: $(date)"
echo "========================================"

# Forward all sbatch script arguments to the Python script
uv run benchmarks/benchmark_repeats.py "$@"

echo "========================================"
echo "  Finished: $(date)"
echo "========================================"
