#!/bin/bash
#SBATCH --job-name=offline_training
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --cpus-per-task=12
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low-unlimit
#SBATCH --account=default-account
#SBATCH --comment=verify_obs_integrity
#SBATCH --qos=debug
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp

pwd
echo "Beginning standard mdp"


python offline_training.py \
    settings.num_grad_steps=300000 \
    agent.mask_ratio=[0.95]