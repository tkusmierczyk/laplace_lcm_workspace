#!/bin/bash
#SBATCH --job-name=lcm-setup
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=logs/%x-%j.out
# Env builds belong in a job on the `cpu` partition, never on the submit-only gateway.
set -euo pipefail
export PROJ="$HOME/laplace_lcm"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR" "$PROJ/logs"
SIF=/shared/sets/singularity/pytorch_24.12-py3.sif
hostname
singularity exec -B "$HOME:$HOME" "$SIF" bash "$PROJ/cluster/setup_env.sh"
