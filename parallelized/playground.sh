#!/bin/bash
#SBATCH --job-name=mjx_testing         # Job name updated for BC
#SBATCH --mail-type=NONE      # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --output=logs/%x-%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

# --- Environment setup ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp_mjx
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export PATH=/usr/local/cuda/bin:$PATH
# ================================================

pwd

python parallelized/playground.py