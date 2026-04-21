#!/bin/bash
# Setup script for Music Genre Classification project
# Creates a virtual environment and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Music Genre Classification - Environment Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing ML dependencies..."
pip install -r "$SCRIPT_DIR/ml/requirements.txt"

echo ""
echo "=== Setup Complete ==="
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To train all models, run:"
echo "  python ml/train_all_models.py"
