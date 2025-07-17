import pandas as pd
import argparse
from utils import check

parser = argparse.ArgumentParser(
    description="Make a nice, cleaned up version of the multijail responses."
)

parser.add_argument(
    ("--input_file", "-i"),
    type=list,
    required=False,
    help="List of paths to input files containing multijail responses.",
)
parser.add_argument(
    ("--output_file", "-o"),
    type=str,
    required=False,
    help="Path to output file for cleaned multijail responses.",
)

args = parser.parse_args()

print(args)
print(check(args))
