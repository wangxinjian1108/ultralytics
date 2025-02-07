#!/bin/bash

# Exit on error
# set -e

# YAML file path
ENV_YAML="conda_env_config.yaml"

# Check if the YAML file exists
if [ ! -f "$ENV_YAML" ]; then
    echo "⚠️ The YAML file '$ENV_YAML' does not exist. Please check the file path."
    exit 1
fi

# Parse YAML to get environment name, Python version, and CUDA version
ENV_NAME=$(grep '^name:' "$ENV_YAML" | awk '{print $2}')
CUDA_VERSION=$(grep 'cuda=' "$ENV_YAML" | awk -F '=' '{print $2}')
PYTHON_VERSION=$(grep 'python=' "$ENV_YAML" | awk -F '=' '{print $2}')

echo "📝 Parsed YAML: Environment Name='$ENV_NAME', Python Version='$PYTHON_VERSION', CUDA Version='$CUDA_VERSION'"

# Function to check if Conda is installed
check_conda() {
    if command -v conda &>/dev/null; then
        echo "✅ Conda is already installed."
        return 0
    else
        echo "⚠️ Conda is not installed."
        return 1
    fi
}

# Function to install Miniconda
install_miniconda() {
    echo "📥 Downloading and installing Miniconda..."
    MINICONDA_INSTALLER="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    INSTALLER="miniconda.sh"

    curl -fsSL "$MINICONDA_INSTALLER" -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$HOME/miniconda"
    rm "$INSTALLER"

    # Initialize Conda
    eval "$HOME/miniconda/bin/conda init"
    echo "✅ Miniconda installation complete. Restart your terminal to use Conda!"
}

# Function to create Conda environment from YAML
create_env_from_yaml() {
    if conda env list | grep -q "^$ENV_NAME\s"; then
        echo "⚡ Conda environment '$ENV_NAME' already exists. Skipping creation."
        return
    fi

    echo "📦 Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION, CUDA $CUDA_VERSION)"
    conda env create -f "$ENV_YAML"
    echo "✅ Conda environment '$ENV_NAME' created successfully!"
}

# Check if Conda is installed, install if not
if ! check_conda; then
    install_miniconda
    # Initialize Conda in the current session
    source "$HOME/miniconda/bin/activate"
fi

source "$HOME/miniconda/bin/activate"
# Create Conda environment from YAML
create_env_from_yaml

# Activate the environment and verify installation
echo "📦 Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Check PyTorch and CUDA availability
echo "🚀 Verifying PyTorch installation..."
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo "✅ Environment '$ENV_NAME' is ready!"
