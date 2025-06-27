#!/bin/bash

# Exit on any error
set -e

# ---- CONFIG ----
ENV_DIR=".venv"
REQ_FILE="requirements.txt"
KERNEL_NAME="myenv"
NOTEBOOK="prefold_automated.ipynb"
PYTHON_BIN="python3"

# ---- CREATE VENV ----
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv $ENV_DIR
fi

# ---- ACTIVATE VENV ----
source $ENV_DIR/bin/activate

# ---- INSTALL DEPENDENCIES ----
echo "Installing requirements..."
pip install --upgrade pip
pip install -r $REQ_FILE
pip install ipykernel jupyterlab

# ---- REGISTER JUPYTER KERNEL ----
echo "Registering Jupyter kernel '$KERNEL_NAME'..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

# ---- LAUNCH JUPYTER WITH TARGET NOTEBOOK ----
echo "Launching JupyterLab with $NOTEBOOK..."
jupyter lab "$NOTEBOOK"