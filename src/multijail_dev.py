"""Evaluate normal and steered models on global mmlu using llm eval harness."""

### give longer max new token generated, sometimes hard to find what exactly the model would have answered.

# Setup
import os
import sys
import datetime
import subprocess
import time
import argparse
import ast

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
    parser.add_argument("--task", required=True, choices=["multijail","mmlu", "or_bench"], type=str, help="The task to evaluate on. Choices are: multijail, mmlu, or_bench.")
    parser.add_argument("--model", required=True, default="meta-llama/meta-llama-3-8b-instruct", type=str, help="The model to evaluate.")

    # Output path and wandb
    parser.add_argument("--output_path", default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results", type=str, help="The output path for results.")
    parser.add_argument("--wandb_project", default="bachelorarbeit", type=str, help="The wandb project name.")

    # Steering arguments
    parser.add_argument("--steering_direction_path", default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/code/direction_llama3_8b.pt", type=str, help="Path to the steering direction .pt file.")
    parser.add_argument(
        "--steering_strengths",
        nargs="*",
        type=float,
        default=None,
        help="List of steering strengths (floats). Can be empty, one or more values."
    )
    parser.add_argument("--steering_layer", type=int, help="Layer number to apply steering to.")

    # Device and other parameters
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to run evaluation on (e.g., cuda:0, cpu).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of evaluation samples (default: None for full dataset).")

    return parser.parse_args()

# Adding the args as variables here is somewhat of a personal preference, makes the script a bit more readable imo.
args = get_args()
run_name = args.run_name
task = args.task
model = args.model
output_path = args.output_path
wandb_project = args.wandb_project
steering_direction_path = args.steering_direction_path
steering_strengths = args.steering_strengths
steering_layer = args.steering_layer
device = args.device
limit = args.limit

# Map mmlu arg to lang specific mmlu tasks for lm_eval. Currently runs English, German, Chinese, and Bengali.
if task == "mmlu":
    task = ",".join(["global_mmlu_en", "global_mmlu_de","global_mmlu_zh", "global_mmlu_bn"])


# Login to weights and biases, huggingface
wandb.login()
with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
    hf_token = f.read().strip()
login(token=hf_token)

# Wandb logging
wandb_user = wandb.api.default_entity
print(f"=== Wandb Information ===")
print(f"Logged in as: {wandb_user}")
print(f"Project: {wandb_project}\n")

# Model run parameters
model = "meta-llama/meta-llama-3-8b-instruct"
eval_tasks = "multijail"

device = "cuda:0"
out_path = "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/multijail"
wandb_project = "bachelorarbeit"
run_name_baseline = "multijail"

# Steering parameters
steering_direction_path = "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/code/direction_llama3_8b.pt"
steering_strengths = [0.33, 0.66, 1.0]
steering_layer = 12

# Load steering components
direction = torch.load(steering_direction_path)
zeros_bias = torch.zeros(direction.shape)

# Failsafe: Ensure output path exists, create if missing.
create_or_ensure_output_path(out_path)

def create_steering_config(layer_num, strength):
    """Create steering config for given layer and strength"""
    return {
        f"layers.{layer_num}": {
            "steering_vector": direction,
            "bias": zeros_bias,
            "steering_coefficient": strength,
            "action": "add",
        }
    }

def run_evaluation(model_type, model_args, run_name):
    """Run lm_eval with given parameters"""

    cmd = [
        "lm_eval",
        "--model",
        model_type,
        "--model_args",
        model_args,
        "--tasks",
        eval_tasks,
        "--output_path",
        f"{out_path}",
        "--device",
        device,
        "--batch_size",
        "auto",
        "--apply_chat_template",
        "--seed",
        "0,1234,1234",  # seeds for python's random, numpy and torch respectively
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
