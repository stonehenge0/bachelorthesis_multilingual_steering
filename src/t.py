# Test script for debugging lm eval runs.

# run test with: 
#python t.py   --run_name "test_run"   --task "mmlu"   --model "meta-llama/meta-llama-3-8b-instruct"   --out_path "/tmp/results"   --wandb_project "test_project"   --steering_direction_path "direction_llama3_8b.pt"   --steering_strengths 0.1 0.5 1.0

import ast
import argparse
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Callable
from copy import deepcopy
import datetime

import torch
from utils import check, seed_everything, create_or_ensure_output_path


# Our base config undergoes three transformations: 
# 1. create_config_globals() -> sets global parameters that are shared across all runs (seed, model, device, etc.)
# 2. create_config_tasks() -> sets task-specific parameters (tasks, apply_chat_template, etc.)
# 3. create_config_steered() -> sets steered model parameters (steering layer, steering strength, etc.)

# In the end we will have n different configs that are each one lm_eval run with n = tasks * steering_strengths +1 (the +1 is for unsteered)


# Argparser
def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    
    parser = argparse.ArgumentParser()

    # Task and model
    parser.add_argument("--run_name", required=True, default="unnamed_run", type=str, help="The run name. It is used for wandb and in naming all output files.")
    parser.add_argument("--task", required=True, choices=["multijail","mmlu", "or_bench"], type=str, help="The task to evaluate on. Choices are: multijail, mmlu, or_bench.")
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

STEER_DIRECTION = torch.load(STEERING_DIRECTION_PATH,map_location=torch.device('cpu'))
ZEROS_BIAS = torch.zeros(STEER_DIRECTION.shape)

MMLU_SUBTASKS_LANGS = ",".join(["global_mmlu_en", "global_mmlu_de","global_mmlu_zh", "global_mmlu_bn"]) # Langs to run MMLU on.
CONFIG_FILEPATH = "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/tmp/"+ str(datetime.datetime.now())  #### check if the str is okay here

# Seeds for everything *in this script*. Lm eval gets the seed over its command line args.
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
        
        # Optional parameters
        if self.apply_chat_template:
            cmd.append("--apply_chat_template")
        
        if self.predict_only:
            cmd.append("--predict_only")
        
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

print("=== Globals Config ===")
print(create_base_config())
print()


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

print("=== Task specific Config ===")
print(create_task_config(create_base_config(), "global_mmlu"))
print()
print(create_task_config(create_base_config(), "multijail"))
print()
print(create_task_config(create_base_config(), "or_bench"))
print()

mmlu_task_config = create_task_config(create_base_config(), "global_mmlu")

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


print("=== Steer specific Config ===")
print(create_steering_config(mmlu_task_config, 0.1))
print()
