"""Evaluate normal and steered models on global mmlu using llm eval harness."""

# Setup
import os
import sys
import wandb
import torch
import subprocess
from lm_eval.loggers import WandbLogger
from lm_eval.utils import sanitize_model_name
from huggingface_hub import login
import time
import datetime

time_start_script = time.time()

# Login wandb and hf
wandb.login()
with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
    hf_token = f.read().strip()
login(token=hf_token)

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

# Failsafe in case output path does not exist
if not os.path.exists(out_path):
    print(f"WARNING: Output path does not exist: {out_path}")
    os.makedirs(out_path, exist_ok=True)
    print(f"Created output directory: {out_path}")
    print(f"Your results will be saved to: {os.path.abspath(out_path)}")

# Wandb logging
wandb_user = wandb.api.default_entity
print(f"=== Wandb Information ===")
print(f"Logged in as: {wandb_user}")
print(f"Project: {wandb_project}\n")


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


time_start_baseline = time.time()

# Run baseline (unsteered) model.
print("=== Running Baseline Model ===")
baseline_success = run_evaluation(
    model_type="hf",
    model_args=f"pretrained={model}",
    run_name=run_name_baseline + "_unsteered",
)

time_start_steered = time.time()
# Run steered model for each steering strength.
print("\n=== Running Steered Models ===")
steered_results = {}

for strength in steering_strengths:

    # steer config for that specific steering strength
    steer_config = create_steering_config(steering_layer, strength)
    print(f"=== Steering Configs ===")
    print()

    output_model_dir_name = sanitize_model_name(
        model
    )  # Lm eval creates a new dir with the sanitized model name to put the outputs. We need to fetch the name to write our config to the right place.

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

time_script_finished = time.time()

# Many many log prints and summary of the run.
print("\n=== CompletedProcess Outputs ===")
print(f"Baseline:\n\t{baseline_success}")
print(f"\nSteered:\\n\t{steered_success}")


print("\n=== Script timing summary ===")
start_dt = datetime.datetime.fromtimestamp(time_start_script)
end_dt = datetime.datetime.fromtimestamp(time_script_finished)
print(f"Start time : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End time   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
total_minutes = (time_script_finished - time_start_script) / 60
print(f"Total time : {total_minutes:.2f} minutes")

print("\nTime breakdown:")
print(
    f"time_start_script ➜ time_start_baseline: {time_start_baseline - time_start_script:.2f} seconds"
)
print(
    f"time_start_baseline ➜ time_start_steered: {time_start_steered - time_start_baseline:.2f} seconds"
)
print(
    f"time_start_steered ➜ time_script_finished: {time_script_finished - time_start_steered:.2f} seconds"
)


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
