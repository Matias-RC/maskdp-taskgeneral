#!/bin/bash
#SBATCH --job-name=exorl_eval_walker_run_small
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fivillagran@uc.cl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
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
echo "Evaluating ExORL walker_run on goal_reaching..."

# === Define algorithms and splits ===
algorithms=("diayn" "disagreement" "icm_apt" "icm"
            "proto" "random" "rnd" "scratch" "smm")

splits=(100 75 50 25 10 5 2 1)

for alg in "${algorithms[@]}"; do
    for split in "${splits[@]}"; do
        padded_split=$(printf "%03d" $split)
        name="${alg}_${padded_split}"


        # zero-pad split for the name
        padded_split=$(printf "%03d" $split)
        name="${alg}_${padded_split}"

        echo "Running evaluation for $name ..."
        python exorl_multieval_finetune_singlegoal.py \
            agent=mdp_goal \
            agent.batch_size=384 \
            seed=3 \
            num_eval_episodes=300 \
            task=walker_run \
            algorithm=$name \
            snapshot_base_dir=/home/bibarel/workspace/finetuned_models_small/ \
            goal_buffer_dir=/home/bibarel/workspace/maskdp_data/maskdp_eval/expert \
            project=exorl-eval-single-goal-finetuned-all \
            replan=true \
            use_wandb=True

        echo "Finished evaluation for $name"
        echo "-----------------------------------"
    done
done
