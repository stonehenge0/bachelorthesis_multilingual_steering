#!/usr/bin/env python3
"""Standalone script to run and evaluate baseline and steered models on different tasks using lm_eval."""

import argparse
import os
import sys
import datetime
import subprocess
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import torch


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_or_ensure_output_path(path: str):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================================
# CONFIG CLASS
# ============================================================================


@dataclass
class Config:
    """Configuration for the evaluation pipeline."""

    model_path: str
    model_alias: str
    steer_type: str
    steering_folder: str
    steering_vector_path: str
    steering_layer: int
    steering_token_position: int
    steering_strengths: List[float]
    debug: bool
    device: str

    @staticmethod
    def artifact_path(cfg):
        """Generate artifact path based on configuration."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            "artifacts",
            f"{cfg.model_alias}",
            f"layer_{cfg.steering_layer}",
            f"token_{cfg.steering_token_position}",
            timestamp,
        )
        return path


# ============================================================================
# EVAL CONFIG
# ============================================================================


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
    limit: Optional[int] = None
    gen_kwargs: str = None

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

        if self.gen_kwargs:
            cmd.extend(["--gen_kwargs", self.gen_kwargs])

        return cmd

    def save_json(self, filepath: str):
        """Save config to a JSON file."""
        import json

        config_dict = {
            "model_type": self.model_type,
            "model_args": self.model_args,
            "tasks": self.tasks,
            "device": self.device,
            "batch_size": self.batch_size,
            "out_path": self.out_path,
            "seed": self.seed,
            "run_name": self.run_name,
            "apply_chat_template": self.apply_chat_template,
            "predict_only": self.predict_only,
            "log_samples": self.log_samples,
            "limit": self.limit,
            "gen_kwargs": self.gen_kwargs,
        }
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def create_base_config(
    MODEL_NAME, MODEL_PATH, DEVICE, OUT_PATH, SEED, LIMIT
) -> EvalConfig:
    """Create base configuration with global settings."""
    config_globals = EvalConfig(
        run_name=f"{MODEL_NAME}",
        model_type="hf",
        model_args=f"pretrained={MODEL_PATH}",
        tasks="",
        device=DEVICE,
        batch_size="auto",
        out_path=OUT_PATH,
        seed=f"{SEED},{SEED},{SEED}",
        limit=LIMIT,
        gen_kwargs="temperature=0.0,max_new_tokens=300,do_sample=False",
    )
    return config_globals


def create_task_config(
    base_config_with_globals, task, mmlu_subtasks_langs=None
) -> EvalConfig:
    """Create task-specific configuration by extending the base config."""
    config = deepcopy(base_config_with_globals)
    config.run_name = f"{task}"
    config.out_path = f"{config.out_path}_{task}"
    print(f"Out path after task config: {config.out_path}")

    if task == "multijail":
        config.tasks = "multijail"
        config.apply_chat_template = True
        config.predict_only = True

    elif task == "global_mmlu":
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

    out_path = os.path.join(f"{out_path}_{config.tasks}", "")

    try:
        config.save_json(filepath=f"{out_path}{config.run_name}.json")
    except Exception as e:
        print(f"Warning: Could not save config for {config.run_name}: {e}")


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
    """Run evaluation on baseline and steered models across multiple tasks."""
    # Constants and setup
    TASKS = ["multijail", "global_mmlu", "or_bench"]
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

    # Load steering components
    STEER_DIRECTION = torch.load(STEER_VECTOR_PATH)
    ZEROS_BIAS = torch.zeros(STEER_DIRECTION.shape)

    all_configs = []

    for task in TASKS:
        # 1. Base config with globals
        base_config = create_base_config(
            MODEL_ALIAS, MODEL_PATH, DEVICE, OUT_PATH, SEED, LIMIT
        )

        # 2. task specific task config
        if task == "global_mmlu":
            task_config = create_task_config(
                base_config, task, mmlu_subtasks_langs=MMLU_SUBTASKS_LANGS
            )
        else:
            task_config = create_task_config(base_config, task)

        all_configs.append(task_config)

        # 3. for steered runs: Steer config and override default arguments
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


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--steer_type",
        choices=["layer_wise", "single_token"],
        type=str,
        required=True,
        help="Type of steering action. Either layer_wise or single_token.",
    )

    parser.add_argument(
        "--steering_folder",
        type=str,
        default="",
        help="Path to the folder with results from steering vector extraction.",
    )

    parser.add_argument(
        "--steering_vector_path",
        type=str,
        required=True,
        help="Path to the steering vector .pt file.",
    )

    parser.add_argument(
        "--steering_layer",
        type=int,
        required=True,
        help="Layer to apply steering to.",
    )

    parser.add_argument(
        "--steering_token_position",
        type=int,
        required=True,
        help="Token position for steering (can be negative for indexing from end).",
    )

    parser.add_argument(
        "--steering_strengths",
        nargs="+",
        type=float,
        required=True,
        help="Steering strengths to use, seperated by a space.",
    )

    parser.add_argument(
        "--device", required=False, type=str, default="cuda:0", help="Device to use."
    )

    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        help="Run on a subsample of dataset and tasks.",
    )

    return parser.parse_args()


def run_pipeline(model_path, args):
    """Run the full pipeline."""

    model_alias = os.path.basename(model_path)

    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        steer_type=args.steer_type,
        steering_folder=args.steering_folder,
        steering_vector_path=args.steering_vector_path,
        steering_layer=args.steering_layer,
        steering_token_position=args.steering_token_position,
        steering_strengths=args.steering_strengths,
        debug=args.debug,
        device=args.device,
    )

    artifact_path = Config.artifact_path(cfg)
    create_or_ensure_output_path(artifact_path)

    results = lm_eval_steered_and_baseline_tasks(
        STEERING_STRENGTHS=cfg.steering_strengths,
        MODEL_ALIAS=cfg.model_alias,
        MODEL_PATH=cfg.model_path,
        STEER_TYPE=cfg.steer_type,
        STEER_VECTOR_PATH=cfg.steering_vector_path,
        STEER_LAYER=cfg.steering_layer,
        TOKEN_POS=cfg.steering_token_position,
        DEVICE=cfg.device,
        DEBUG=cfg.debug,
        ARTIFACT_PATH=artifact_path,
    )

    print(f"Configuration: {cfg}")
    print(f"Artifact path: {artifact_path}")
    print(f"Results: {results}")

    return results


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, args=args)
