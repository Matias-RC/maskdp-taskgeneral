#!/bin/bash
#SBATCH --job-name=PPO_Finetune          # Job name
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
echo "Pretraining PPO."

export PYTHONWARNINGS="ignore"
export HYDRA_FULL_ERROR=1 
python finetune_ppo.py \
    task=walker_run \
    algorithm=icm \
    resume_dir=/home/bibarel/workspace/exorl_models/output/2025.10.23/033220_mdp/snapshot/walker/1/icm \
    seed=1 \
    agent=mdp \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    use_wandb=false