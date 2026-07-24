#!/bin/bash
#SBATCH --job-name=mdp_clip_sweep      # Base job name
#SBATCH --output=logs/%x/%j.out        # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x/%j.err         # Error log                    
#SBATCH --gres=gpu:4                   # Requesting 4 GPUs for parallel jobs
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --nodelist=yodaxico
#SBATCH --partition=ialab
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Start directly in your project folder
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

# --- Environment setup ---
source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp-ppo

pwd
echo "Beginning mdp clip_coef sweep across 4 GPUs"

# Define clip_coef hyperparameter values
CLIP_COEFS=(0.08 0.12 0.16 0.2)

# Track background job PIDs
PIDS=()
GPU_ID=0

# Loop through all 4 clip coefficient values
for clip_coef in "${CLIP_COEFS[@]}"; do
    
    RUN_NAME="mdp_clip_coef_${clip_coef}"
    
    echo "Launching run: ${RUN_NAME} on GPU ${GPU_ID}"
    
    # Assign run to a distinct GPU (0, 1, 2, 3) and set clip_coef
    CUDA_VISIBLE_DEVICES=${GPU_ID} python online_training.py \
        ++env_cls_route.truncation_limit=512 \
        ++agent.transformer_cfg.clip_coef=${clip_coef} \
        ++agent.name="${RUN_NAME}" \
        exp_name="${RUN_NAME}" \
        project=mdp_ppo_new_gae \
        num_grad_steps=1000 \
        notes="sweep_clip_coef_${clip_coef}" &
        
    PIDS+=($!)
    
    # Increment GPU index for the next job
    ((GPU_ID++))
done

echo "All 4 clip_coef runs active on GPUs 0-3. Waiting for completion..."

# Block until all background tasks complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All clip_coef sweep jobs completed successfully!"