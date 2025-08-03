"""Run and evaluate baseline and steered models on different tasks using lm_eval."""

#---------------------
# Code logic:
# Every combination of steered/unsteered model and task is run as a separate lm_eval run. This is necessary because of the combinations of flags.
# The run is based on the EvalConfig dataclass which is converted to command line arguments for lm_eval.

# Our base config undergoes three transformations: 
# 1. create_config_globals() -> sets global parameters that are shared across all runs (seed, model, device, etc.)
# 2. create_task_config() -> sets task-specific parameters (tasks, apply_chat_template, etc.)
# 3. create_steering_config() -> sets steered model parameters (steering layer, steering strength, etc.)

# In the end we will have n different configs that are each one lm_eval run with n = tasks * steering_strengths +1 (the +1 is for unsteered)
# ---------------------

### give longer max new token generated, sometimes hard to find what exactly the model would have answered.
### Currently working on saving and retrieving the the steer config in temp, might work though.

# Setup
import os
import sys
import datetime
import subprocess
import time
import argparse
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from copy import deepcopy

import wandb
import torch
from lm_eval.loggers import WandbLogger
from lm_eval.utils import sanitize_model_name
from huggingface_hub import login

from utils import check, seed_everything, create_or_ensure_output_path

# Argparser
def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    
    parser = argparse.ArgumentParser()

    # Task and model
    parser.add_argument("--run_name", required=True, default="unnamed_run", type=str, help="The run name. It is used for wandb and in naming all output files.")
    parser.add_argument("--task", required=True, choices=["multijail","global_mmlu", "or_bench"], type=str, help="The task to evaluate on. Choices are: multijail, mmlu, or_bench.")
    parser.add_argument("--model", required=True, default="meta-llama/meta-llama-3-8b-instruct", type=str, help="The model to evaluate.")

    # Output path and wandb
    parser.add_argument("--out_path", default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results", type=str, help="The output path for results.")
    parser.add_argument("--wandb_project", default="bachelorarbeit", type=str, help="The wandb project name. bachelorarbeit project per default.")

    # Steering arguments
    parser.add_argument("--steering_direction_path", default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/code/direction_llama3_8b.pt", type=str, help="Path to the steering direction .pt file.")
    parser.add_argument(
        "--steering_strengths",
        nargs="*",
        type=float,
        default=None,
        help="Any number of steering strength seperated by a space: --steering_strengths 0.1 0.5 1.0"
    )
    parser.add_argument("--steering_layer", type=int, help="Layer number to apply steering to.")

    # Device and other parameters
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to run evaluation on (e.g., cuda:0, cpu).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of evaluation samples (default: None for full dataset).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")

    return parser.parse_args()

# Adding the args as variables here is somewhat of a personal preference, makes the script a bit more readable imo.
args = get_args()
RUN_NAME = args.run_name
TASK = args.task
MODEL = args.model
OUT_PATH = args.out_path
WANDB_PROJECT = args.wandb_project
STEERING_DIRECTION_PATH = args.steering_direction_path
STEERING_STRENGTHS = args.steering_strengths
STEERING_LAYER = args.steering_layer
DEVICE = args.device
LIMIT = args.limit
SEED = args.seed

MMLU_SUBTASKS_LANGS = ",".join(["global_mmlu_en", "global_mmlu_de","global_mmlu_zh", "global_mmlu_bn"]) # Langs to run MMLU on.
CONFIG_FILEPATH = "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/tmp/"+ str(datetime.datetime.now())  #### check if the str is okay here

# Failsafe: Ensure output path exists, create if missing.
create_or_ensure_output_path(OUT_PATH)

# Seeds for everything *in this script*. Lm eval gets the seed over its command line args.
seed_everything(SEED)

# Login wandb and huggingface
wandb.login()
with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
    hf_token = f.read().strip()
login(token=hf_token)

# Wandb logging
wandb_user = wandb.api.default_entity
print(f"=== Wandb Information ===")
print(f"Logged in as: {wandb_user}")
print(f"Project: {WANDB_PROJECT}\n")

# Load steering components
STEER_DIRECTION = torch.load(STEERING_DIRECTION_PATH)
ZEROS_BIAS = torch.zeros(STEER_DIRECTION.shape)

# Failsafe: Ensure output path exists, create if missing.
create_or_ensure_output_path(OUT_PATH)

@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""
    # Required fields (no defaults)
    model_type: str
    model_args: str
    tasks: str
    device: str
    batch_size: str
    out_path: str
    seed: str
    
    # Optional fields (with defaults)
    run_name: str = ""
    apply_chat_template: bool = False #### Should they be T/F or just go to None here? Same for others like limit flag etc. 
    predict_only: bool = False
    log_samples: bool = True
    wandb_args: Optional[str] = None
    limit: Optional[int] = None
    
    def to_cmd_args(self) -> List[str]:
        """Convert config to command line arguments for lm_eval."""
        # Required parameters
        cmd = [
            "lm_eval",
            "--model", self.model_type,
            "--model_args", self.model_args,
            "--tasks", self.tasks,
            "--output_path", self.out_path,
            "--device", self.device,
            "--batch_size", self.batch_size,
            "--seed", self.seed,
            "--log_samples",
        ]
        
        # Optional flag parameters (no values needed)
        if self.apply_chat_template:
            cmd.append("--apply_chat_template")
        
        if self.predict_only:
            cmd.append("--predict_only")
        
        # Optional parameters with values
        if self.limit:
            cmd.extend(["--limit", str(self.limit)])
        
        if self.wandb_args:
            cmd.extend(["--wandb_args", self.wandb_args])
        
        return cmd

# 1. Base config with globals
def create_base_config() -> EvalConfig:
    """Create base configuration with global settings."""

    config_globals = EvalConfig(
        run_name="_".join([RUN_NAME, MODEL]),
        model_type="hf",  # Default, will be overridden for steered
        model_args=f"pretrained={MODEL}",
        tasks="",     
        device=DEVICE,
        batch_size="auto",
        out_path=OUT_PATH,
        seed=f"{SEED},{SEED},{SEED}", # seeds for python's random, numpy and torch respectively",
        wandb_args=f"project={WANDB_PROJECT}",  # Base wandb config, run name will be added later
    )    

    return config_globals

# 2. Task-specific configurations
def create_task_config(base_config_with_globals, task) -> EvalConfig:
    """
    Create task-specific configuration by extending the base config.
    """
    config = deepcopy(base_config_with_globals)
    config.run_name = f"{config.run_name}_{task}"

    if task == "multijail":
        config.tasks = "multijail"
        config.apply_chat_template = True
        config.predict_only = True
    
    elif task == "global_mmlu":
        config.tasks = "global_mmlu_en,global_mmlu_de,global_mmlu_zh,global_mmlu_bn"
        config.apply_chat_template = False
        config.predict_only = False
    
    elif task == "or_bench":
        config.tasks = "or_bench"
        config.apply_chat_template = True
        config.predict_only = True
    
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks are: multijail, global_mmlu, or_bench.")
    
    return config

# 3. Steer specific configurations
def create_steering_config(task_specific_config,steer_strength):
    """Create steering config for given layer and strength"""

    config = deepcopy(task_specific_config)
    config.run_name = f"{config.run_name}_L{STEERING_LAYER}_S{steer_strength}"

    # steering config as input to lm_eval
    steer_config_parameter = {
        f"layers.{STEERING_LAYER}": {
            "steering_vector": STEER_DIRECTION,
            "bias": ZEROS_BIAS,
            "steering_coefficient": steer_strength,
            "action": "add",
        }
    }

    # it's a bit annoying that lm_eval expects the steering config as a filepath, but we have to save it to a file and then read it back.
    torch.save(steer_config_parameter, CONFIG_FILEPATH)

    config.model_type = "steered"
    config.model_args = f"pretrained={MODEL},steer_path={CONFIG_FILEPATH}"

    return config

def run_evaluation(model_type, model_args, run_name):
    """Run lm_eval with given parameters"""

    cmd = [
        "lm_eval",
        "--model",
        model_type,
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--output_path",
        f"{out_path}",
        "--device",
        device,
        "--batch_size",
        "auto",
        "--apply_chat_template",
        "--seed",
        "1234,1234,1234",  # seeds for python's random, numpy and torch respectively
        "--predict_only",  # Predict only, since we will evaluate with our own llm judge later.
        "--wandb_args",
        f"project={wandb_project},name={run_name}",
        "--log_samples",
    ]

    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    return result

# Run baseline (unsteered) model.
print("=== Running Baseline Model ===")
baseline_success = run_evaluation(
    model_type="hf",
    model_args=f"pretrained={model}",
    run_name=run_name_baseline + "_unsteered",
)

# Run steered model for each steering strength.
print("\n=== Running Steered Models ===")
steered_results = {}

for strength in steering_strengths:

    # steer config
    steer_config = create_steering_config(steering_layer, strength)
    print(f"=== Steering Configs ===")
    print()

    output_model_dir_name = sanitize_model_name(
        model
    ) 


    config_filename = (
        run_name_baseline + f"_steer_config_layer{steering_layer}_strength{strength}.pt"
    )
    config_filepath = os.path.join(out_path, output_model_dir_name, config_filename)

    torch.save(steer_config, config_filepath)

    # Run eval
    run_name = f"steered_layer{steering_layer}_strength{strength}"
    steered_success = run_evaluation(
        model_type="steered",
        model_args=f"pretrained={model},steer_path={config_filepath}",
        run_name=run_name,
    )

    steered_results[strength] = steered_success

# Many many log prints and summary of the run.
print("\n=== CompletedProcess Outputs ===")
print(f"Baseline:\n\t{baseline_success}")
print(f"\nSteered:\\n\t{steered_success}")

print("\n=== Results Summary ===")
baseline_no_error = True
steered_no_error = True

if "Error" in str(baseline_success):
    baseline_no_error = False

if "Error" in str(steered_success):
    steered_no_error = False

print(f"Baseline: {'Success' if baseline_success else 'Failed'}")
print("Steered models:")
for strength, steered_success in steered_results.items():
    print(f"  Strength {strength}: {'Success' if steered_success else 'Failed'}")
