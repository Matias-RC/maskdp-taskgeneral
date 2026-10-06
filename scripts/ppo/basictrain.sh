#!/bin/bash
#SBATCH --job-name=walker_ppo     # Job name
#SBATCH --output=logs/%x/%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x/%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Start directly in your project folder
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

# --- Environment setup ---
# Use absolute path to guarantee conda sources correctly
source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp-ppo

pwd
echo "Beginning ppo mdp in walker task // Easy"

CLIP_COEF=0.16
ENTROPY_COEF=0.001
ACTION_REPEAT=4
RUN_NAME="walker_clip_${CLIP_COEF}_ent_${ENTROPY_COEF}"

python online_training.py \
    ++env_cls_route.action_repeat=4 \
    ++agent.transformer_cfg.clip_coef=0.16 \
    ++agent.transformer_cfg.entropy_coef=0.01 \
    ++agent.name=walker \
    exp_name=walker \
    project=mdp_ppo_xqc \
    num_grad_steps=200 \