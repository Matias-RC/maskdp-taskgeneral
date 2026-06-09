#!/bin/bash
#SBATCH --job-name=maskdp_eval_walker_dataset       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matias.rodriguez@cenia.cl    # El mail del usuario
#SBATCH --output=logs/%x-%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=16               # CPU cores
#SBATCH --nodelist=scylla
#SBATCH --partition=ialab
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Start directly in your project folder
#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

# --- Environment setup ---
# Use absolute path to guarantee conda sources correctly
source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp

pwd
echo "Evaluating walker dataset..."


python eval_goal.py \
    agent=mdp_goal \
    agent.batch_size=384 \
    seed=45856 \
    num_eval_episodes=3 \
    task=walker_walk \
    snapshot_base_dir=snapshot \
    goal_buffer_dir=/home/matias_rodriguez/maskdp-taskgeneral/maskdp_data/maskdp_train/walker/expert \
    snapshot_ts=0 \
    project=eval-single-goal \
    replan=false \
    use_wandb=True \
    +exp_name=eval_dataset \
    ++inspect_data_only=True \
    ++inspect_max_steps=1000