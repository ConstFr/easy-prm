#!/bin/bash
#SBATCH --job-name=easy-prm-bert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/easy-prm-bert-%j.out
#SBATCH --error=logs/easy-prm-bert-%j.err

set -euo pipefail

PROJECT_ROOT="/home/dzianis.piatrashyn/university/optimization/easy-prm"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

source ~/.bashrc
conda activate easy_prm

bash run_training_bert.sh "$@"
