#! /bin/bash
#SBATCH --job-name=proxy_2days_aya-translation-long-oldT07gen-run
#SBATCH -c 8 
#SBATCH --mem 40G  
#SBATCH -p scc-gpu 
#SBATCH -t 47:00:00 
#SBATCH -G A100:1
#SBATCH --output=./slurm_files/slurm-%x-%j.out     
#SBATCH --error=./slurm_files/slurm-%x-%j.err 

# GWDG changed network setup, so need to set here 
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"

# Printing out some info on filepaths.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Python and torch info. Uncomment for debugging
python --version
export PYTHONPATH=/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering:$PYTHONPATH
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

# Run script.
python pipeline/subfns/translate_to_english.py \
    --model "CohereLabs/aya-101" \
    --out_path "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/new_runs_27ter_2days" \
    --folderpath "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/pipeline/Llama-8B-t0.7" \




    