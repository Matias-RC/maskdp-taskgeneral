#!/bin/bash
#SBATCH --job-name=mdp_baseline      # Job name
#SBATCH --output=logs/%x/%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x/%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=20             # CPU cores
#SBATCH --nodelist=llaima
#SBATCH --partition=ialab
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Start directly in your project folder
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

# --- Environment setup ---
# Use absolute path to guarantee conda sources correctly
source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp

pwd
echo "Beginning standard mdp"


python offline_training.py \
    project=baselines\
    settings.num_grad_steps=300010 \
    agent=mlp_baseline.yaml