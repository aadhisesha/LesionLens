#!/usr/bin/env bash
# Run Baseline TransAttUNet experiment (Phase 2)
set -e

CONFIG=${1:-"configs/baseline.yaml"}
echo "Running Baseline TransAttUNet with config: $CONFIG"
python training/train.py --config "$CONFIG"
