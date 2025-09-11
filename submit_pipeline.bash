#! /bin/bash
#SBATCH --job-name=70b-llama
#SBATCH -c 8 #
#SBATCH -p scc-gpu 
#SBATCH -t 07:00:00 
#SBATCH --G=H100:1
#SBATCH --output=./slurm_files/slurm-%x-%j.out     
#SBATCH --error=./slurm_files/slurm-%x-%j.err  

source ~/.bashrc 

# Printing out some info on filepaths.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Python and torch info. Uncomment for debugging
python --version
# python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# Conda environment setup.
source /mnt/vast-standard/home/stein65/u14374/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/vast-standard/home/stein65/u14374/miniforge3/envs/modified_lm_eval_env
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

export PYTHONPATH=/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering:$PYTHONPATH

# Run script
python pipeline/run_pipeline.py\
  --model_path "meta-llama/Llama-3.1-70B-Instruct" \
  --steer_type "layer_wise" \
  --steering_folder "" \
  --steering_vector_path "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/data/steer_data_Llama-3.1-70B-Instruct/direction.pt" \
  --steering_layer 35 \
  --steering_token_position -5 \
  --steering_strengths 0.33 0.66 1.0 \
  --device "cuda:0" \

# Scan HF cache
echo "HF cache:"
hf cache scan -v