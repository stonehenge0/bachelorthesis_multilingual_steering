"""Evaluate normal and steered models on global mmlu using llm eval harness."""

# Setup
import os
import sys
import wandb
import torch
import subprocess
from lm_eval.loggers import WandbLogger
from huggingface_hub import login

# Login wandb and hf
wandb.login()

with open(os.path.expanduser('~/.cache/huggingface/token'), 'r') as f:
    hf_token = f.read().strip()
login(token=hf_token)

# Model run parameters
model = "meta-llama/meta-llama-3-8b-instruct"
eval_tasks = ",".join(["global_mmlu_en", "global_mmlu_de","global_mmlu_zh", "global_mmlu_bn"])

device = "cuda:0"
out_path = "/scratch1/users/u14374/bachelorarbeit"
wandb_project = "bachelorarbeit"
run_name_baseline = "global_mmlu"

# Steering parameters
steering_direction_path = "/scratch1/users/u14374/bachelorarbeit/direction.pt"
steering_strengths = [0.33,0.66, 1.0] 
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

def create_steering_config(layer_num, strength):
    """Create steering config for given layer and strength"""
    return {
        f"layers.{layer_num}": {
            "steering_vector": direction,
            "bias": zeros_bias,
            "steering_coefficient": strength,
            "action": "add"
        }
    }

def run_evaluation(model_type, model_args, run_name):
    """Run lm_eval with given parameters"""
    
    cmd = [
        "lm_eval",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", eval_tasks,
        "--output_path", f"{out_path}",
        "--device", device,
        "--batch_size", "auto",
        "--seed", "42",
        "--wandb_args", f"project={wandb_project},name={run_name}",
        "--log_samples"
    ]
    
    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    sys.stdout.write(f"Running: {run_name}")
    sys.stdout.write(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    return result


# Run baseline (unsteered) model.
print("=== Running Baseline Model ===")
baseline_success = run_evaluation(
    model_type="hf",
    model_args=f"pretrained={model}",
    run_name= run_name_baseline + "_unsteered"
)

# Run steered model for each steering strength.
print("\n=== Running Steered Models ===")
steered_results = {}

for strength in steering_strengths:
    
    # steer config for that specific steering strength
    steer_config = create_steering_config(steering_layer, strength)
    config_filename = run_name_baseline + f"steer_config_layer{steering_layer}_strength{strength}.pt"
    torch.save(steer_config, config_filename)
    
    # Run eval
    run_name = f"steered_layer{steering_layer}_strength{strength}"
    steered_success = run_evaluation(
        model_type="steered",
        model_args=f"pretrained={model},steer_path={config_filename}",
        run_name=run_name
    )
    
    steered_results[strength] = steered_success


# Print output summary
print("\n=== Results Summary ===")
print(f"Baseline: {'Success' if baseline_success else 'Failed'}")
print("Steered models:")
for strength, steered_success in steered_results.items():
    print(f"  Strength {strength}: {'Success' if steered_success else 'Failed'}")
