#!/usr/bin/env bash
#
# setup_directories.sh
# Creates the required data and logging directory structure for the bioinformatics pipeline.

set -euo pipefail

echo "[INFO] Initializing project directory structure..."

# Define all required directories based on the pipeline's I/O requirements
DIRECTORIES=(
    "data/raw/genome/star_index"
    "data/raw/epigenome"
    "data/raw/eclip"
    "data/raw/rbp"
    "data/raw/models/rnafm"
    "data/indices/STAR_hg38"
    "data/prepped"
    "data/features"
    "data/models"
    "data/results"
    "data/lineage"
    "data/figures"
    "logs"
)

# Create directories iteratively, ignoring existing ones
for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
    echo "[CREATED] $dir"
done

echo "[INFO] Directory structure successfully initialized."