import argparse
import os

from config import Config
from lm_eval_steered_and_baseline_tasks import (
    lm_eval_steered_and_baseline_tasks,
)
from utils import create_or_ensure_output_path

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
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


def run_pipeline(model_path):
    """Run the full pipeline."""

    model_alias = os.path.basename(model_path)

    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        steering_vector_path=args.steering_vector_path,
        steering_layer=args.steering_layer,
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
        STEER_VECTOR_PATH=cfg.steering_vector_path,
        STEER_LAYER=cfg.steering_layer,
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
    run_pipeline(model_path=args.model_path)
