#! /bin/bash
#SBATCH --job-name=qwen_judge_t0
#SBATCH -c 8 #
#SBATCH --mem 40G  
#SBATCH -p scc-gpu 
#SBATCH -t 06:00:00 
#SBATCH -G A100:1 
#SBATCH --constraint=inet
#SBATCH --output=./slurm_files/slurm-%x-%j.out     
#SBATCH --error=./slurm_files/slurm-%x-%j.err      

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

# Run script.

# qwen: /mnt/ceph-hdd/projects/ag_gipp/emma/hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
# aya101: /mnt/ceph-hdd/projects/ag_gipp/emma/hf_cache/hub/models--CohereLabs--aya-101/snapshots/231cff3a9729ccdaee18839b32aaabac5278a21c

python pipeline/subfns/llm_judge.py \
 --model "Qwen/Qwen3-14B" \
 --out_path "..." \
 --files_to_process_dict '{
  "multijail_baseline": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/pipeline/runs/Llama-3.1-8B-Instruct_multijail/meta-llama__Llama-3.1-8B-Instruct/samples_multijail_2025-09-25T22-33-47.630820.jsonl",
  "multijail_L11_S0.33": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/pipeline/runs/Llama-3.1-8B-Instruct_multijail_L11_S0.33/meta-llama__Llama-3.1-8B-Instruct/samples_multijail_2025-09-25T23-48-46.673186.jsonl",
  "multijail_L11_S0.66": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/pipeline/runs/Llama-3.1-8B-Instruct_multijail_L11_S0.66/meta-llama__Llama-3.1-8B-Instruct/samples_multijail_2025-09-26T01-16-10.055377.jsonl",
  "multijail_L11_S1.0": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/pipeline/runs/Llama-3.1-8B-Instruct_multijail_L11_S1.0/meta-llama__Llama-3.1-8B-Instruct/samples_multijail_2025-09-26T03-11-53.025882.jsonl"
}'

