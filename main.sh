#!/usr/bin/env bash
#
# main.sh
# Master Orchestration Script for Bioinformatics Pipeline

set -euo pipefail

# --- Logging Configuration ---
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ORCHESTRATOR] $1"
}

err() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# --- Step Definitions ---
run_extract_l1() {
    log "Running Step 1: L1 RMSK Extraction..."
    bash extract_l1_rmsk.sh
}

run_epigenetic_filter() {
    log "Running Step 2: Epigenetic Filter..."
    python epigenetic_filter.py
}

run_chimeric_mapping() {
    log "Running Step 3: Chimeric Mapping & EM Deconvolution..."
    bash chimeric_mapping.sh
}

run_feature_extraction() {
    log "Running Step 4: Hybrid Feature Extraction..."
    python feature_extraction.py
}

run_model_training() {
    log "Running Step 5: XGBoost Model Training..."
    # Note: explicit paths are passed because this script requires them
    python model_training.py \
        --features data/features/hybrid_matrix.parquet \
        --peaks data/prepped/ground_truth_ago2_peaks.parquet
}

run_statistical_validation() {
    log "Running Step 6: Statistical Validation & FDR Control..."
    python statistical_validation.py
}

run_all() {
    run_extract_l1
    run_epigenetic_filter
    run_chimeric_mapping
    run_feature_extraction
    run_model_training
    run_statistical_validation
    log "Complete pipeline executed successfully."
}

# --- Argument Parsing ---
if [[ "$#" -eq 0 ]]; then
    echo "Usage: ./main.sh [--all] [--step <step_name>]"
    echo "Steps: extract_l1, epigenetic_filter, chimeric_mapping, feature_extraction, model_training, statistical_validation"
    exit 1
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all)
            run_all
            exit 0
            ;;
        --step)
            STEP_NAME="$2"
            case $STEP_NAME in
                extract_l1) run_extract_l1 ;;
                epigenetic_filter) run_epigenetic_filter ;;
                chimeric_mapping) run_chimeric_mapping ;;
                feature_extraction) run_feature_extraction ;;
                model_training) run_model_training ;;
                statistical_validation) run_statistical_validation ;;
                *) err "Unknown step: ${STEP_NAME}"; exit 1 ;;
            esac
            shift
            ;;
        *)
            err "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done