import os
import random
import numpy as np
import torch
import pandas as pd

# Debugging tensors.
def check(x, name_of_x=False):
    """Helper function for checking shapes and types during debugging."""
    print("====CHECKING======")
    if name_of_x:
        print(f"name:{name_of_x}")
    print(f"type: {type(x)}")

    if hasattr(x, "shape"):
        print(f"shape: {x.shape}")

    if hasattr(x, "len"):
        print(f"Length: {len(x)}")
    print("====END OF CHECK======")


# Seeds.
def seed_everything(seed=11711):
    "Set random seeds for python, numpy, and torch"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Create output path if missing.
def create_or_ensure_output_path(path):
    """Create the output directory if it does not exist."""
    if not os.path.exists(path):
        print(f"WARNING: Output path does not exist: {path}")
        
        os.makedirs(path, exist_ok=True)
        print(f"Created output directory: {path}")
        print(f"Your results will be saved to: {os.path.abspath(path)}")

# prints out basic info about a dataframe with many possible parameters to print
def check_df(df, name = None, shape=True, columns=True, info=False, describe=False, NAs=True, check_presence_empty_strings = True, unique=False, head=False):
    """
    Print out selected statistics and information about a pandas DataFrame.
    Set each argument to True to print that df functionality.
    """
    if name: 
        print(f"\n\033[1mDataFrame: {name}\033[0m")
    if shape:
        print("\n\033[4mDataFrame Shape\033[0m:")
        print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    if columns:
        print("\n\033[4mDataFrame Columns\033[0m:")
        print(df.columns.tolist())
    if info:
        print("\n\033[4mDataFrame Info\033[0m:")
        print(df.info())
    if describe:
        print("\n\033[4mDataFrame Description\033[0m:")
        print(df.describe(include='all').to_string())
    if NAs:
        print("\n\033[4mChecking for NAs\033[0m:")
        na_counts = df.isna().sum()
        if na_counts.any():
            print("\n\033[1mNAs found!\033[0m:")
            for col, count in na_counts.items():
                if count > 0:
                    print(f"  In Column {col}: count: {count}")
        else:
            print("\n\033[1mNo NAs present in any column.\033[0m")
    if unique:
        print("\n\033[4mUnique Values per Column (should be 0)\033[0m:")
        for col in df.columns:
            print(f"  {col}: {df[col].nunique()}")
    if head:
        print("\n\033[4mDataFrame Head (first 5 rows)\033[0m:")
        print(df.head().to_string())
    