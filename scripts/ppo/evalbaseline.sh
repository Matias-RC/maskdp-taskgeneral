#!/bin/bash
#SBATCH --job-name=eval_walker_ppo     # Job name
#SBATCH --output=logs/%x/%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x/%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=12             # CPU cores
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low
#SBATCH --account=default-account          
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --mem=10G
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


python baseline_eval_temp.py \
    env_cls_route=gym_single \
    agent=mlp_baseline \
    trainer=baselineppo \
    exp_name=walker \
    project=mlp_ppo_xqc \
    num_grad_steps=1000 \
    transparent_training=false \