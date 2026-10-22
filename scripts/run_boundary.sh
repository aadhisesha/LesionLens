#!/usr/bin/env bash
# Run Boundary-Aware TransAttUNet experiment (Phase 4)
set -e

CONFIG=${1:-"configs/boundary.yaml"}
echo "Running Boundary-Aware TransAttUNet with config: $CONFIG"
python training/train.py --config "$CONFIG"
