#! /bin/bash
#SBATCH --job-name=pipe_test
#SBATCH -c 8 #
#SBATCH --mem 80G  
#SBATCH -p scc-gpu 
#SBATCH -t 1:00:00 
#SBATCH -G A100:1 
#SBATCH --output=../slurm_files/slurm-%x-%j.out     
#SBATCH --error=../slurm_files/slurm-%x-%j.err      

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


# Run script
python run_pipeline.py\
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --steer_type "layer_wise" \
  --steering_folder "" \
  --steering_vector_path "C:\Users\emste\Documents\cloned_Gits\bachelorthesis_multilingual_steering\data\Llama-3.1-8B-Instruct-direction.pt" \
  --steering_layer 11 \
  --steering_token_position -2 \
  --steering_strengths 0.5 1.0 \
  --device "cuda:0" \
  --debug True

# Clean up HF cache.
echo "HF cache before deletion:"
hf cache scan -v

echo "Cleaning up HF cache..."
rm -rf ~/.cache/huggingface/hub/models--*
echo "Cache cleanup completed"

echo "HF cache after deletion:"
hf cache scan -v