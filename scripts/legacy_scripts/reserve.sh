#!/bin/bash
#SBATCH --job-name=sfinetune          # Job name
#SBATCH --mail-type=NONE                 # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fivillagran@uc.cl    # El mail del usuario
#SBATCH --output=logs/%x-%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --chdir=/home/bibarel/workspace

# --- Environment setup ---
source "./miniconda3/etc/profile.d/conda.sh"
conda activate maskdp-ppo
cd "./maskdp-taskgeneral"
pwd
echo "Doing stuff..."

python donotreadme.py
