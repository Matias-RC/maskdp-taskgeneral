#!/bin/bash
#SBATCH --job-name=maskdp_bc_walker_walk         # Job name updated for BC
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zzdude70@gmail.com    # El mail del usuario
#SBATCH --output=logs/%x-%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --nodelist=scylla
#SBATCH --partition=ialab
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

# --- Environment setup ---
source "./miniconda3/etc/profile.d/conda.sh"
conda activate maskdp
pwd
echo "Fine-tuning MaskDP (Behavioral Cloning) on expert walker_walk task..."

PRETRAINED_SNAPSHOT="/home/matias_rodriguez/maskdp-taskgeneral/snapshot/walker/2/snapshot_200000.pt"

python behavioral_cloning.py \
    task=walker_walk \
    pretrained_path=$PRETRAINED_SNAPSHOT \
    +agent.freeze_backbone=false \
    agent.PE_type=joint\
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    batch_size=256 \
    lr=1e-4 \
    num_grad_steps=100000 \
    project=final_mt_mdp_bc \
    use_wandb=True \
    seed=1