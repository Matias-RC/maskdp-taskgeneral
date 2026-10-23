#!/bin/bash
#SBATCH --job-name=maskdp_adapter_test
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --cpus-per-task=12
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low-unlimit
#SBATCH --account=default-account
#SBATCH --comment=check_pretrained_maskdp_adapter_as_ppo_actor_critic_walker_walk
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
# Extra hydra overrides are passed through, e.g.
#   sbatch scripts/testing/maskdp_adapter_test.sh trainer_cfg.model.fresh_heads=true
TOTAL_STEPS=${TOTAL_STEPS:-20480}
SAVE_PATH=${SAVE_PATH:-snapshots/walker_walk/ppo_maskdp_smoke/model.pt}

echo "Smoke test: transformer PPO with the pretrained MaskDP adapter on DMC walker_walk ($TOTAL_STEPS steps) $*"
python -u transformer_training.py \
        use_wandb=${USE_WANDB:-true} \
        settings=walker_walk_maskdp \
        trainer_cfg.total_steps=$TOTAL_STEPS \
        trainer_cfg.save_path=$SAVE_PATH \
        "$@"
