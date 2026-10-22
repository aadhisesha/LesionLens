#!/usr/bin/env bash
# Run Efficient TransAttUNet experiment (Phase 3)
set -e

CONFIG=${1:-"configs/efficient.yaml"}
echo "Running Efficient TransAttUNet with config: $CONFIG"
python training/train.py --config "$CONFIG"
