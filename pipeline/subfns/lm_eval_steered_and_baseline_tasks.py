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

### ToDos:
### Naming of WandB runs not there yet.
### Steer type arguments not used yet, to be implemented.

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

from .utils import seed_everything, create_or_ensure_output_path


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
        self.out_path = os.path.join(self.out_path)
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

        # Optional flags without values
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


def create_base_config(
    MODEL_NAME, MODEL_PATH, DEVICE, OUT_PATH, SEED, WANDB_PROJECT, LIMIT
) -> EvalConfig:
    """Create base configuration with global settings."""
    config_globals = EvalConfig(
        run_name=f"{MODEL_NAME}",
        model_type="hf",  # Default, will be overridden for steered
        model_args=f"pretrained={MODEL_PATH}",
        tasks="",
        device=DEVICE,
        batch_size="auto",
        out_path=OUT_PATH,
        seed=f"{SEED},{SEED},{SEED}",  # Three seeds for python's random, numpy and torch respectively",
        wandb_args=f"project={WANDB_PROJECT}",  # Base wandb config, run name will be added later
        limit=LIMIT,
    )

    return config_globals


def create_task_config(
    base_config_with_globals, task, mmlu_subtasks_langs=None
) -> EvalConfig:
    """
    Create task-specific configuration by extending the base config.
    """
    config = deepcopy(base_config_with_globals)
    config.run_name = f"{task}"
    config.out_path = f"{config.out_path}_{task}"
    print(f"Out path after task config: {config.out_path}")

    if task == "multijail":
        config.tasks = "multijail"
        config.apply_chat_template = True
        config.predict_only = True

    elif task == "global_mmlu":
        # Use provided subtasks if available
        if mmlu_subtasks_langs:
            config.tasks = mmlu_subtasks_langs
        else:
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


def create_steering_config(
    task_specific_config,
    steer_strength,
    STEER_LAYER,
    STEER_DIRECTION,
    ZEROS_BIAS,
    CONFIG_FILEPATH,
    MODEL_PATH,
):
    """Create steering config for given layer and strength"""
    config = deepcopy(task_specific_config)
    config.run_name = f"{config.run_name}_L{STEER_LAYER}_S{steer_strength}"
    config.out_path = f"{config.out_path}_L{STEER_LAYER}_S{steer_strength}"
    print(f"Out path after steer config: {config.out_path}")

    steer_config_parameter = {
        f"layers.{STEER_LAYER}": {
            "steering_vector": STEER_DIRECTION,
            "bias": ZEROS_BIAS,
            "steering_coefficient": steer_strength,
            "action": "add",
        }
    }
    torch.save(steer_config_parameter, CONFIG_FILEPATH)

    config.model_type = "steered"
    config.model_args = f"pretrained={MODEL_PATH},steer_path={CONFIG_FILEPATH}"
    return config


def run_and_save(config: EvalConfig, out_path: str):
    """Run lm_eval with the given configuration and save outputs."""

    create_or_ensure_output_path(config.out_path)
    cmd = config.to_cmd_args()
    print(f"Running command for {config.run_name}:\n {' '.join(cmd)}")

    out = subprocess.run(cmd, capture_output=True, text=True)

    if out.returncode != 0:
        raise RuntimeError(
            f"Error running command for {config.run_name}:\n{out.stderr}\n Returncode:{out.returncode}"
        )

    out_path = os.path.join(
        f"{out_path}_{config.tasks}", ""
    )


    try:
        config.save_json(
            filepath=f"{out_path}{config.run_name}.json"
        ) 
    except Exception as e:
        print(f"Warning: Could not save config for {config.run_name}: {e}")


all_configs = []


def lm_eval_steered_and_baseline_tasks(
    STEERING_STRENGTHS,
    MODEL_ALIAS,
    MODEL_PATH,
    STEER_TYPE,
    STEER_VECTOR_PATH,
    STEER_LAYER,
    TOKEN_POS,
    DEVICE,
    DEBUG,
    ARTIFACT_PATH,
):
    # Constants and setup
    TASKS = ["multijail", "global_mmlu", "or_bench"]
    WANDB_PROJECT = "bachelorarbeit"
    SEED = 1234
    OUT_PATH = ARTIFACT_PATH

    # Default MMLU subtasks. These are all langs from MMLU that overlap with Or_bench
    MMLU_SUBTASKS_LANGS = ",".join(
        [
            "global_mmlu_en",
            "global_mmlu_ar",
            "global_mmlu_zh",
            "global_mmlu_it",
            "global_mmlu_ko",
        ]
    )

    # Debug with small samples
    LIMIT = 2 if DEBUG else None
    if DEBUG:
        MMLU_SUBTASKS_LANGS = ",".join(
            [
                "global_mmlu_en",
                "global_mmlu_ar",
            ]
        )

    seed_everything(SEED)

    # Login to wandb
    wandb.login()

    wandb_user = wandb.api.default_entity
    print(f"=== Wandb Information ===")
    print(f"Logged in as: {wandb_user}")
    print(f"Project: {WANDB_PROJECT}\n")

    # Load steering components
    STEER_DIRECTION = torch.load(STEER_VECTOR_PATH)
    ZEROS_BIAS = torch.zeros(STEER_DIRECTION.shape)

    all_configs = []

    for task in TASKS:
        # 1. Base config with globals
        base_config = create_base_config(
            MODEL_ALIAS, MODEL_PATH, DEVICE, OUT_PATH, SEED, WANDB_PROJECT, LIMIT
        )

        # 2. task specific task config
        if task == "global_mmlu":
            task_config = create_task_config(
                base_config, task, mmlu_subtasks_langs=MMLU_SUBTASKS_LANGS
            )
        else:
            task_config = create_task_config(base_config, task)

        all_configs.append(task_config)

        # 3. for steered runs: Steer config and override default arguments (model type form hf to steered)
        for strength in STEERING_STRENGTHS:

            CONFIG_FILEPATH = os.path.join(
                ARTIFACT_PATH,
                f"steer_config_{MODEL_ALIAS}_S{strength}_L{STEER_LAYER}.pt",
            )
            steered_config = create_steering_config(
                task_config,
                strength,
                STEER_LAYER,
                STEER_DIRECTION,
                ZEROS_BIAS,
                CONFIG_FILEPATH,
                MODEL_PATH,
            )
            all_configs.append(steered_config)

    for config in all_configs:
        run_and_save(config, OUT_PATH)
