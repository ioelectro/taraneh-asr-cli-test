#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
VENV=".venv"
MODEL_DIR="models/Taraneh-Q0.6B"

echo "== Taraneh-Q0.6B Persian ASR setup (Linux) =="

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON was not found. Install Python 3.12 and python3.12-venv first."
    exit 1
fi

# PortAudio is required by sounddevice on many Linux distributions.
if command -v apt-get >/dev/null 2>&1; then
    echo "Installing PortAudio system libraries..."
    sudo apt-get update
    sudo apt-get install -y libportaudio2 portaudio19-dev
fi

"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "Installing CPU-only PyTorch..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing Python requirements..."
python -m pip install -r requirements.txt

echo "Downloading Taraneh-Q0.6B model..."
mkdir -p "$MODEL_DIR"
hf download areffarhadi/Taraneh-Q0.6B --local-dir "$MODEL_DIR"

echo
echo "Setup complete."
echo "Activate environment: source $VENV/bin/activate"
echo "Run your script:      python speech_recognition.py"
echo "Make sure the script is in this directory and INPUT_DEVICE is correct."
