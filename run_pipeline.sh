#!/usr/bin/env bash
set -e

DATASET=${1:-"XES3G5M"}
CONFIG="configs/macr_xes3g5m.yaml"
CDM_TYPE="ncd"

echo "=========================================================="
echo "  Running MACR End-to-End Pipeline on ${DATASET} (${CDM_TYPE})  "
echo "=========================================================="

if [ -d "data/raw/${DATASET}" ]; then
    echo "--> [Step 0] Preprocessing raw dataset..."
    python scripts/run_phase0_preprocess.py --raw-dir "data/raw/${DATASET}" --output-dir "data/${DATASET}"
fi

echo "--> [Step 1] Metric-Preserving Student Anchoring & Semantic Prior Extraction..."
python scripts/run_phase1_anchoring.py --config "${CONFIG}" --cdm-type "${CDM_TYPE}"

echo "--> [Step 2] Building mechanism-driven contrastive pairs..."
python scripts/build_contrastive_pairs.py --config "${CONFIG}"

echo "--> [Step 2] Building parallel-form variants..."
python scripts/build_parallel_forms.py --config "${CONFIG}" --split test --allow-fallback

echo "--> [Step 3] Training MACR item recalibration model..."
python scripts/train_macr.py --config "${CONFIG}"

echo "--> [Step 4] Single-form evaluation..."
python scripts/evaluate_macr.py --config "${CONFIG}" --split test

echo "--> [Step 4] Parallel-form reliability fusion evaluation..."
python scripts/evaluate_parallel_forms.py --config "${CONFIG}" --split test

echo "=========================================================="
echo "  MACR Pipeline Completed Successfully!                   "
echo "=========================================================="
