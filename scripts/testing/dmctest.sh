#!/bin/bash
#SBATCH --job-name=dmc_ppo_smoke
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --cpus-per-task=12
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low-unlimit
#SBATCH --account=default-account
#SBATCH --comment=smoke_test_dmc_gymnasium
#SBATCH --qos=debug
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp-ppo
set -e

echo "Smoke test: baseline PPO on DMC walker_walk"
python baseline_training.py \
        use_wandb=false \
        settings=walker_walk \
        trainer_cfg.total_steps=4096

echo "Smoke test: transformer PPO on DMC walker_walk"
python transformer_training.py \
        use_wandb=false \
        settings=walker_walk \
        trainer_cfg.total_steps=4096
