#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_quadruped_unsup_data     # Job name
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zzdude70@gmail.com    # El mail del usuario
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

#SBATCH --chdir=/home/matias_rodriguez/maskdp-taskgeneral

source "/home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh"
conda activate maskdp

pwd
echo "Pretraining MaskDP on quadruped unsup..."

python pretrain.py \
    agent=mdpAA_jointPE \
    agent.batch_size=256 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    num_grad_steps=200010 \
    task=quadruped_run \
    snapshot_dir=snapshot \
    resume=false \
    project=pretrain_unsup \
    use_wandb=True \
    seed=2 \
    +is_local_data=false \
    +hf_path=fangchenliu/maskdp_data/maskdp_train/quadruped/unsup/buffer