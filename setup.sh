#!/bin/bash -i
set -e

ENV_NAME="safety_steering"
PYTHON_VERSION="3.10"

module load miniforge3 || true  # load conda module if available

# Function to check if conda is installed
check_conda_installed() {
    if command -v conda &> /dev/null; then
        echo "Conda is available."
    else
        echo "Conda not found. Installing Miniconda locally..."
        install_miniconda
    fi
}

# Function to install Miniconda
install_miniconda() {
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
}

# Function to check/create environment
setup_env() {
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' already exists."
    else
        echo "Creating environment '$ENV_NAME'..."
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    fi
}

# Huggingface Setup
setup_hf() {
    echo "Please enter your Hugging Face token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Creating Hugging Face config directory..."
        mkdir -p ~/.huggingface
        chmod 700 ~/.huggingface

        echo "Installing/Updating Hugging Face Hub (for CLI)..."
        pip install --upgrade --quiet huggingface_hub

        echo "Storing HF token securely..."
        echo "{\"token\": \"$token\"}" > ~/.huggingface/token.json
        chmod 600 ~/.huggingface/token.json

        echo "Logging in to Hugging Face..."
        huggingface-cli login --token "$token"

        echo "Hugging Face setup complete. Token stored in ~/.huggingface/"
    else
        echo "No token entered. Skipping Hugging Face setup."
    fi
}

# Clone and install the modified LLM eval harness
setup_eval_harness() {
    if [ ! -d "modified_llm_eval_harness" ]; then
        echo "Cloning modified_llm_eval_harness..."
        git clone https://github.com/stonehenge0/modified_llm_eval_harness.git
    fi

    echo "Installing modified_llm_eval_harness in editable mode..."
    pip install -e modified_llm_eval_harness
}


# ------------------------
# Main script execution
# ------------------------

# Conda and setup
check_conda_installed
setup_env
setup_hf

# Initialize conda for current shell
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing packages from environment.yml..."
conda env update -n $ENV_NAME -f environment.yml --prune

# Install eval harness
setup_eval_harness

echo "Setup complete. Activate environment with: conda activate $ENV_NAME"
