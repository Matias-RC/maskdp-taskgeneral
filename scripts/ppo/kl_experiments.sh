#!/bin/bash
#SBATCH --job-name=mdp_sweep           # Base job name
#SBATCH --output=logs/%x/%j.out        # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x/%j.err         # Error log                    
#SBATCH --gres=gpu:4                   # Requesting 4 GPUs for parallel jobs
#SBATCH --cpus-per-task=20             # CPU cores
#SBATCH --nodelist=yodaxico
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
source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp-ppo

pwd
echo "Beginning mdp hyperparameter sweep across 4 GPUs"

# Define parameter grids
STDS=(0.25 0.5)
LRS=("1e-4" "5e-5")

# Track background job PIDs
PIDS=()
GPU_ID=0

# Loop through all 4 parameter combinations
for std in "${STDS[@]}"; do
    for lr in "${LRS[@]}"; do
        
        RUN_NAME="mdp_std_${std}_lr_${lr}"
        
        echo "Launching run: ${RUN_NAME} on GPU ${GPU_ID}"
        
        # Assign run to a distinct GPU (0, 1, 2, 3) and use + prefix for dynamic dynamic overrides
        CUDA_VISIBLE_DEVICES=${GPU_ID} python online_training.py \
            ++std=${std} \
            ++agent.lr=${lr} \
            ++agent.name="${RUN_NAME}" \
            exp_name="${RUN_NAME}" \
            project=mdp_ppo_kl_exps \
            notes="sweep_std_${std}_lr_${lr}" &
            
        PIDS+=($!)
        
        # Increment GPU index for the next job
        ((GPU_ID++))
    done
done

echo "All 4 runs active on GPUs 0-3. Waiting for completion..."

# Block until all background tasks complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All sweep jobs completed successfully!"
