#!/bin/bash

# Directory containing slurm files
SLURM_DIR="./slurm_files"

find "$SLURM_DIR" -type f -delete
echo "Cleaned."