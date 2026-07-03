#!/bin/bash
#SBATCH --job-name=finetune_mdp_walker_run       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fivillagran@uc.cl    # El mail del usuario
#SBATCH --output=logs/%x-%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=16               # CPU cores
#SBATCH --nodelist=peteroa
#SBATCH --partition=debug
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --chdir=/home/bibarel/workspace

# --- Environment setup ---
source "./miniconda3/etc/profile.d/conda.sh"
conda activate maskdp
cd "./maskdp-taskgeneral"
pwd
echo "Finetuning on walker_run task..."

for N in 100 75 50 25 10 5 2 1
do
    # zero-pad to 3 digits for the name
    PADDED=$(printf "%03d" $N)

    # Too many workers for a small dataset will give empty arrays
    # to some workers, so cap workers at 2N
    workers=$(( 2*N < 16 ? 2*N : 16 ))

    python finetune.py \
        name=smm_${PADDED} \
        agent=mdp \
        agent.batch_size=384 \
        agent.transformer_cfg.traj_length=64 \
        agent.transformer_cfg.loss="total" \
        agent.transformer_cfg.n_embd=256 \
        agent.transformer_cfg.n_head=4 \
        agent.transformer_cfg.n_enc_layer=3 \
        agent.transformer_cfg.n_dec_layer=2 \
        agent.transformer_cfg.norm='l2' \
        agent.lr=1e-5 \
        num_grad_steps=50010 \
        task=walker_run \
        replay_buffer_dir=/home/bibarel/workspace/finetune \
        data_split=${N} \
        replay_buffer_num_workers=${workers} \
        resume=true \
        resume_dir=/home/bibarel/workspace/exorl_models/output/2025.10.23/061254_mdp/snapshot/walker/1/smm \
        resume_step=400000 \
        project=finetune_mdp_small \
        use_wandb=True \
        seed=1
done