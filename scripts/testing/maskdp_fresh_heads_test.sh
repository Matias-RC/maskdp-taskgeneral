#!/bin/bash
#SBATCH --job-name=maskdp_fresh_heads_test
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --cpus-per-task=12
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low-unlimit
#SBATCH --account=default-account
#SBATCH --comment=check_maskdp_adapter_fresh_heads_on_latent_plus_state_walker_walk
#SBATCH --qos=debug
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp-ppo


# Short run: TOTAL_STEPS env steps (default 20480 = 10 PPO updates of 256 x 8 envs).
TOTAL_STEPS=${TOTAL_STEPS:-20480}

echo "Smoke test: MaskDP adapter with fresh heads on DMC walker_walk ($TOTAL_STEPS steps) $*"
python -u transformer_training.py \
        use_wandb=${USE_WANDB:-true} \
        settings=walker_walk_maskdp \
        trainer_cfg.model.fresh_heads=true \
        trainer_cfg.total_steps=$TOTAL_STEPS \
        trainer_cfg.save_path=snapshots/walker_walk/ppo_maskdp_fresh_smoke/model.pt \
        "$@"
