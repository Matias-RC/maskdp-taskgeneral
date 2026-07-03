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
algorithms=("scratch" "icm" "proto" "scratch_010" "icm_010" "proto_010"
            "diayn_100" "diayn_010"
            "disagreement_100" "disagreement_010" "rnd_100" "rnd_010"
            "icm_apt_100" "icm_apt_010" "smm_100" "smm_010"
            "random_100" "random_010"
            "diayn_050" "disagreement_050" "icm_050" "icm_apt_050"
            "proto_050" "random_050" "rnd_050" "scratch_050" "smm_050"
            "smm_025" "scratch_025" "rnd_025" "random_025" "proto_025"
            "icm_apt_025" "icm_025" "disagreement_025" "diayn_025"
            "smm_075" "scratch_075" "rnd_075" "icm_075" "icm_apt_075"
            "proto_075" "diayn_075" "disagreement_075" "random_075"
            "diayn_001" "disagreement_001" "icm_apt_001" "icm_001"
            "proto_001" "random_001" "rnd_001" "scratch_001" "smm_001")
base_dirs=(
    "2025.10.30/070122_mdp" # Scratch_100
    "2025.10.30/071534_mdp" # ICM_100
    "2025.10.30/071536_mdp" # Proto_100
    "2025.10.30/072951_mdp" # Scratch_010
    "2025.10.30/105857_mdp" # ICM_010
    "2025.10.30/111333_mdp" # Proto_010

    "2025.11.01/030004_mdp" # Diayn_100
    "2025.11.01/030058_mdp" # Diayn_010

    "2025.11.03/150120_mdp" # Disagreement_100
    "2025.11.03/150326_mdp" # Disagreement_010
    "2025.11.03/164852_mdp" # RND_100
    "2025.11.03/164907_mdp" # RND_010
    
    "2025.11.03/184024_mdp" # ICM-APT_100
    "2025.11.03/184045_mdp" # ICM-APT_010
    "2025.11.03/190342_mdp" # SMM_100
    "2025.11.03/190359_mdp" # SMM_010

    "2025.11.03/190702_mdp" # Random_100
    "2025.11.03/204704_mdp" # Random_010

    "2025.11.04/014420_mdp" # Diayn_050
    "2025.11.04/014424_mdp" # Disagreement_050
    "2025.11.04/014431_mdp" # ICM_050
    "2025.11.04/014435_mdp" # ICM-APT_050
    "2025.11.04/014529_mdp" # Proto_050
    "2025.11.04/014538_mdp" # Random_050
    "2025.11.04/014545_mdp" # RND_050
    "2025.11.04/054223_mdp" # Scratch_050
    "2025.11.04/054248_mdp" # SMM_050
    
    "2025.11.04/144309_mdp" # SMM_025
    "2025.11.04/144312_mdp" # Scratch_025
    "2025.11.04/144315_mdp" # RND_025
    "2025.11.04/144324_mdp" # Random_025
    "2025.11.04/144327_mdp" # Proto_025
    "2025.11.04/144332_mdp" # ICM-APT_025
    "2025.11.04/144336_mdp" # ICM_025
    "2025.11.04/184035_mdp" # Disagreement_025
    "2025.11.04/184040_mdp" # Diayn_025

    "2025.11.04/195612_mdp" # SMM_075
    "2025.11.04/195616_mdp" # Scratch_075
    "2025.11.04/205125_mdp" # RND_075
    "2025.11.05/001959_mdp" # ICM_075
    "2025.11.05/002002_mdp" # ICM-APT_075
    "2025.11.05/002007_mdp" # Proto_075
    "2025.11.05/025643_mdp" # Diayn_075
    "2025.11.05/025710_mdp" # Disagreement_075
    "2025.11.11/191529_mdp" # Random_075

    "2025.11.10/071116_mdp" # Diayn_001
    "2025.11.10/071428_mdp" # Disagreement_001
    "2025.11.10/071453_mdp" # ICM-APT_001
    "2025.11.10/155407_mdp" # ICM_001
    "2025.11.10/155414_mdp" # Proto_001
    "2025.11.10/213842_mdp" # Random_001
    "2025.11.10/213846_mdp" # RND_001
    "2025.11.10/213850_mdp" # Scratch_001
    "2025.11.11/014703_mdp" # SMM_001
    
)

# === Loop over all algorithm/base_dir pairs ===
for i in "${!algorithms[@]}"; do
    alg=${algorithms[$i]}
    dir=${base_dirs[$i]}

    echo "Running evaluation for $alg ..."
    python exorl_multieval_finetune_singlegoal.py \
        agent=mdp_goal \
        agent.batch_size=384 \
        seed=3 \
        num_eval_episodes=300 \
        task=walker_run \
        algorithm=$alg \
        snapshot_base_dir=/home/bibarel/workspace/finetuned_models/${dir}/finetunes \
        goal_buffer_dir=/home/bibarel/workspace/maskdp_data/maskdp_eval/expert \
        project=exorl-eval-single-goal-finetuned-all \
        replan=false \
        use_wandb=True

    echo "Finished evaluation for $alg"
    echo "-----------------------------------"
done
