#!/bin/bash
#SBATCH --job-name=maskdp_frozen_backbone_test
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --cpus-per-task=12
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low-unlimit
#SBATCH --account=default-account
#SBATCH --comment=check_maskdp_adapter_with_frozen_backbone_walker_walk
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


PRETRAINED=snapshots/pretrained_mdp/walker/walker_walk_9999_1.pt

echo "Smoke test: transformer PPO with the pretrained MaskDP adapter, frozen backbone, on DMC walker_walk ($TOTAL_STEPS steps) $*"
python -u transformer_training.py \
        use_wandb=true \
        settings=walker_walk_maskdp_frozen \
        trainer_cfg.model.fresh_heads=true \
        trainer_cfg.save_path=snapshots/walker_walk/ppo_maskdp_fresh_smoke/model.pt \
        "$@"

