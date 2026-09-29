#!/bin/bash
# ==============================================================================
# run_mac.sh -- 7B Code Generation Pipeline for Apple Silicon Mac (M1/M2/M3/M4)
# ==============================================================================
set -e

echo "--------------------------------------------------------"
echo "  Starting Confidence-Gated Verification 7B Generation"
echo "--------------------------------------------------------"

# 1. Environment Setup
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment (venv)..."
    python3 -m venv venv
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio transformers accelerate

# 2. Run Generation (50 tasks = 500 completions)
echo "[4/4] Launching Qwen2.5-Coder-7B-Instruct on Apple Silicon GPU (MPS)..."
echo "      (Progress is auto-saved after every task. Safe to interrupt and resume)"
python src/generate.py --model 7B --limit 50

echo "--------------------------------------------------------"
echo "  Generation complete!"
echo "  Pushing generations to GitHub..."
echo "--------------------------------------------------------"

git add data/generations/full_7B
git commit -m "Add Qwen2.5-Coder-7B generations for 50 benchmark tasks" || true
git push origin main

echo "--------------------------------------------------------"
echo "  Successfully pushed to GitHub!"
echo "  All done. Just tell your teammate to run 'git pull origin main'."
echo "--------------------------------------------------------"

