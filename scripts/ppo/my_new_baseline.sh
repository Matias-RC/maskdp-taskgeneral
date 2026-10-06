#!/bin/bash
#SBATCH --job-name=baseline_expr    # Job name
#SBATCH --output=logs/%x/%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x/%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=12             # CPU cores
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab-low-unlimit
#SBATCH --account=default-account          
#SBATCH --qos=debug
#SBATCH --time=1:00:00
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


python online_training.py \
    env_cls_route=gymsmojoco \
    agent=mlpbaseline2 \
    trainer=baseline2ppo \
    exp_name=walker \
    project=mlpbaseline2 \
    num_grad_steps=100 \