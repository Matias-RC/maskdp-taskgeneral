#!/bin/bash
#SBATCH --job-name=exorl_eval_walker_run_all
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

# === Define algorithms and base dirs ===
algorithms=("proto")
base_dirs=("2025.10.30/071536_mdp")

# === Loop over all algorithm/base_dir pairs ===
for i in "${!algorithms[@]}"; do
    alg=${algorithms[$i]}
    dir=${base_dirs[$i]}

    # Default snapshot_ts
    snapshot_ts=800000

    echo "Running evaluation for $alg ..."
    python exorl_eval_finetune_singlegoal.py \
        agent=mdp_goal \
        agent.batch_size=384 \
        seed=3 \
        num_eval_episodes=300 \
        task=walker_run \
        algorithm=$alg \
        snapshot_base_dir=/home/bibarel/workspace/finetuned_models/${dir}/finetunes \
        goal_buffer_dir=/home/bibarel/workspace/maskdp_data/maskdp_eval/expert \
        snapshot_ts=$snapshot_ts \
        project=exorl-eval-single-goal-finetuned-test \
        replan=false \
        use_wandb=True

    echo "Finished evaluation for $alg"
    echo "-----------------------------------"
done
