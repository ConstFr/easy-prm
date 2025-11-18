#!/bin/bash
#SBATCH --job-name=easy-prm-llama
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=225G
#SBATCH --time=12:00:00
#SBATCH --output=logs/easy-prm-bert-%j.out
#SBATCH --error=logs/easy-prm-bert-%j.err

set -euo pipefail

PROJECT_ROOT="/home/dzianis.piatrashyn/university/optimization/easy-prm"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

source ~/.bashrc
conda activate easy_prm

bash run_training_llama_base.sh
