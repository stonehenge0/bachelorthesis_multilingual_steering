"""Run and evaluate baseline and steered models on different tasks using lm_eval."""

# ---------------------
# Code logic:
# Every combination of steered/unsteered model and task is run as a separate lm_eval run. This is necessary because of the combinations of flags.
# The run is based on the EvalConfig dataclass which is converted to command line arguments for lm_eval.

# Our base config undergoes three transformations:
# 1. create_config_globals() -> sets global parameters that are shared across all runs (seed, model, device, etc.)
# 2. create_task_config() -> sets task-specific parameters (tasks, apply_chat_template, etc.)
# 3. create_steering_config() -> sets steered model parameters (steering layer, steering strength, etc.)

# In the end we will have n different configs that are each one lm_eval run with n = tasks * steering_strengths +1 (the +1 is for unsteered)
# ---------------------

### Options for sampling
### Naming of WandB runs not there yet.

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
    parser.add_argument(
        "--run_name",
        required=True,
        default="unnamed_run",
        type=str,
        help="The run name. It is used for wandb and in naming all output files.",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        nargs="+",
        choices=["multijail", "global_mmlu", "or_bench"],
        type=str,
        help="One or more tasks to evaluate on. Choices are: multijail, global_mmlu, or_bench. Example: --task multijail global_mmlu",
    )
    parser.add_argument(
        "--model",
        required=True,
        default="meta-llama/meta-llama-3-8b-instruct",
        type=str,
        help="The model to evaluate.",
    )

    # Output path and wandb
    parser.add_argument(
        "--out_path",
        default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/",
        type=str,
        help="The output path for results.",
    )
    parser.add_argument(
        "--wandb_project",
        default="bachelorarbeit",
        type=str,
        help="The wandb project name. bachelorarbeit project per default.",
    )

    # Steering arguments
    parser.add_argument(
        "--steering_direction_path",
        default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/code/direction_llama3_8b.pt",
        type=str,
        help="Path to the steering direction .pt file.",
    )
    parser.add_argument(
        "--steering_strengths",
        nargs="*",
        type=float,
        default=None,
        help="Any number of steering strength seperated by a space: --steering_strengths 0.1 0.5 1.0",
    )
    parser.add_argument(
        "--steering_layer", required=True, type=int, help="Layer to apply steering to."
    )

    # Device and other parameters
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to run evaluation on (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation samples (default: None for full dataset).",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )

    return parser.parse_args()


# Adding the args as variables here is somewhat of a personal preference, makes the script a bit more readable imo.
args = get_args()

RUN_NAME = args.run_name
TASKS = args.tasks
MODEL = args.model
OUT_PATH = args.out_path
WANDB_PROJECT = args.wandb_project
STEERING_DIRECTION_PATH = args.steering_direction_path
STEERING_STRENGTHS = args.steering_strengths
STEERING_LAYER = args.steering_layer
DEVICE = args.device
LIMIT = args.limit
SEED = args.seed

MMLU_SUBTASKS_LANGS = ",".join(
    ["global_mmlu_en", "global_mmlu_de", "global_mmlu_zh", "global_mmlu_bn"]
)  # Langs to run MMLU on.
CONFIG_FILEPATH = f"/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/tmp/steer_config_{RUN_NAME}.pt"

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
    apply_chat_template: bool = False
    predict_only: bool = False
    log_samples: bool = True
    wandb_args: Optional[str] = None
    limit: Optional[int] = None

    def to_cmd_args(self) -> List[str]:
        """Convert config to command line arguments for lm_eval."""
        # Required parameters

        self.out_path = os.path.join(self.out_path, self.run_name)  # Better file naming
        cmd = [
            "lm_eval",
            "--model",
            self.model_type,
            "--model_args",
            self.model_args,
            "--tasks",
            self.tasks,
            "--output_path",
            self.out_path,
            "--device",
            self.device,
            "--batch_size",
            self.batch_size,
            "--seed",
            self.seed,
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

    @staticmethod
    def save_json(self, filepath: str):
        """Save config to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())


# 1. Base config with globals
def create_base_config() -> EvalConfig:
    """Create base configuration with global settings."""

    config_globals = EvalConfig(
        run_name=f"{MODEL}",
        model_type="hf",  # Default, will be overridden for steered
        model_args=f"pretrained={MODEL}",
        tasks="",
        device=DEVICE,
        batch_size="auto",
        out_path=OUT_PATH,
        seed=f"{SEED},{SEED},{SEED}",  # seeds for python's random, numpy and torch respectively",
        wandb_args=f"project={WANDB_PROJECT}",  # Base wandb config, run name will be added later
        limit=LIMIT,
    )

    return config_globals


# 2. Task-specific configurations
def create_task_config(base_config_with_globals, task) -> EvalConfig:
    """
    Create task-specific configuration by extending the base config.
    """
    config = deepcopy(base_config_with_globals)
    config.run_name = f"{task}"  # f"{config.run_name}_{task}"

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
        raise ValueError(
            f"Unknown task: {task}. Supported tasks are: multijail, global_mmlu, or_bench."
        )

    return config


# 3. Steer specific configurations
def create_steering_config(task_specific_config, steer_strength):
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


def run_and_save(config: EvalConfig):
    """Run lm_eval with the given configuration and print the command."""

    # Run command
    cmd = config.to_cmd_args()
    print(f"Running command for {config.run_name}:\n {' '.join(cmd)}")
    out = subprocess.run(cmd, capture_output=True, text=True)

    # Check if Command ran successfully
    if out.returncode != 0:
        raise RuntimeError(
            f"Error running command for {config.run_name}:\n{out.stderr}\n Returncode:{out.returncode}"
        )

    # Save config to JSON in output path.
    try:
        config.save_json(f"{OUT_PATH}{config.run_name}.json")
    except:
        print(f"Warning: Could not save config for {config.run_name}")


all_configs = []


def main():
    # baseline run and task config
    for task in TASKS:
        base_config = create_base_config()
        task_config = create_task_config(base_config, task)
        all_configs.append(task_config)

        # Steer specific runs
        for strength in STEERING_STRENGTHS:
            steered_config = create_steering_config(task_config, strength)
            all_configs.append(steered_config)

    for config in all_configs:
        run_and_save(config)


if __name__ == "__main__":
    main()
