#!/bin/bash

# Exit on any error
set -e

# ---- CONFIGURATION ----
ENV_DIR=".venv"
REQ_FILE="requirements.txt"
ENTRY_SCRIPT="main.py"  # or whatever your script is
PYTHON_BIN="python3"    # or specify full path if needed

# ---- CREATE VENV ----
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv $ENV_DIR
fi

# ---- ACTIVATE VENV ----
source $ENV_DIR/bin/activate

# ---- UPGRADE PIP AND INSTALL DEPENDENCIES ----
echo "Installing dependencies from $REQ_FILE..."
pip install --upgrade pip
pip install -r $REQ_FILE

# ---- RUN YOUR SCRIPT ----
echo "Running $ENTRY_SCRIPT..."
python $ENTRY_SCRIPT