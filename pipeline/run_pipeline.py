import argparse
import os

from submodules.test import test
from submodules.test import print_something, return_doubled

# from pipeline.config import Config


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
        required=True,
        help="Path to the folder with results from steering vector extraction.",
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

    return parser.parse_args()


# test single function call out of script
def doubled_values(value: int) -> int:
    return return_doubled(value)


# test calling full script
def run_test():
    text, d_val = test("schän", 6)
    return text, d_val


def run_pipeline(model_path):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    # cfg = Config(model_alias=model_alias, model_path=model_path)

    return model_alias


if __name__ == "__main__":
    # args = parse_arguments()
    # run_pipeline(model_path=args.model_path)
    t_res = test(text="schän", value=6)
    print(t_res)
