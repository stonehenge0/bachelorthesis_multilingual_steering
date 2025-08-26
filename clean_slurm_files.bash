#!/bin/bash

# Directory containing slurm files
SLURM_DIR="../slurm_files"

echo "Cleaned."
find "$SLURM_DIR" -type f -delete