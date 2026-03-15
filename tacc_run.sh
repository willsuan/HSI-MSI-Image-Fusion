#!/bin/bash
#
# GRSS Segmentation Training — TACC Vista
# Run this script from the HSI-MSI-Image-Fusion directory on Vista.
#
# Usage:
#   1. Copy/clone repo to Vista
#   2. Download phase2.zip from http://hyperspectral.ee.uh.edu/QZ23es1aMPH/2018IEEE/phase2.zip
#      (you may need to email sprasad2@uh.edu for access first)
#   3. Place phase2.zip in this directory
#   4. Run: bash tacc_run.sh setup      (one-time: installs deps + organizes data)
#   5. Run: sbatch tacc_run.sh train     (submits SLURM job for all 3 models)
#   6. After job completes: bash tacc_run.sh results
#
# -------------------------------------------------------------------
# SLURM directives (used when submitted via sbatch)
# -------------------------------------------------------------------
#SBATCH -J grss_train
#SBATCH -o grss_train_%j.out
#SBATCH -e grss_train_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A IRI23021

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-help}"

# -------------------------------------------------------------------
# SETUP: install dependencies + organize data
# -------------------------------------------------------------------
if [ "$MODE" = "setup" ]; then
    echo "=== Setting up environment ==="

    # Load TACC modules
    module load cuda/12.4 2>/dev/null || module load cuda/11.8 2>/dev/null || true
    module load python3/3.11 2>/dev/null || module load python3 2>/dev/null || true
    module load conda 2>/dev/null || true

    # Try conda first, fall back to venv
    if command -v conda &>/dev/null; then
        if ! conda info --envs 2>/dev/null | grep -q "grss"; then
            echo "Creating conda environment 'grss'..."
            conda create -n grss python=3.10 -y
        fi
        eval "$(conda shell.bash hook)"
        conda activate grss
    else
        echo "conda not found, using python venv..."
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
    fi

    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision
    pip install matplotlib scipy scikit-learn einops tqdm torchmetrics spectral \
                rasterio tensorboard ConfigSpace albumentations tifffile pyyaml pandas imgaug

    # Organize data
    echo "=== Organizing GRSS data ==="
    mkdir -p data/GRSS

    # Find any GRSS zip file (phase2.zip or 2018IEEEGRSS_DataFusionChallenge.fixed.zip)
    ZIP_FILE=$(ls phase2.zip 2018IEEEGRSS*.zip 2>/dev/null | head -1)

    if [ -n "$ZIP_FILE" ]; then
        echo "Extracting $ZIP_FILE..."
        unzip -o "$ZIP_FILE" -d data/GRSS/
        echo "Data extracted to data/GRSS/"
    elif [ -d "data/GRSS/ImageryAndTrainingGT" ]; then
        echo "Data already organized."
    else
        echo ""
        echo "ERROR: No GRSS zip found and data/GRSS/ImageryAndTrainingGT doesn't exist."
        echo "Download from: https://utexas.app.box.com/s/32clotuwcsmeutqooe4k48xet4m4ziso"
        echo "Place the zip in this directory and re-run: bash tacc_run.sh setup"
        exit 1
    fi

    # Verify expected files exist
    echo "=== Verifying data files ==="
    HSI_HDR="data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/FullHSIDataset/20170218_UH_CASI_S4_NAD83.hdr"
    GT_TIF="data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif"

    # Search for the files if not at expected path (zip structure may differ)
    if [ ! -f "$HSI_HDR" ]; then
        echo "HSI .hdr not found at expected path. Searching..."
        FOUND_HDR=$(find data/GRSS -name "20170218_UH_CASI_S4_NAD83.hdr" 2>/dev/null | head -1)
        if [ -n "$FOUND_HDR" ]; then
            echo "  Found at: $FOUND_HDR"
            echo "  You may need to adjust data_dir in the config YAMLs."
        else
            echo "  WARNING: HSI .hdr file not found anywhere in data/GRSS/"
        fi
    else
        echo "  HSI .hdr: OK"
    fi

    if [ ! -f "$GT_TIF" ]; then
        echo "GT .tif not found at expected path. Searching..."
        FOUND_GT=$(find data/GRSS -name "2018_IEEE_GRSS_DFC_GT_TR.tif" 2>/dev/null | head -1)
        if [ -n "$FOUND_GT" ]; then
            echo "  Found at: $FOUND_GT"
        else
            echo "  WARNING: GT .tif file not found anywhere in data/GRSS/"
        fi
    else
        echo "  GT .tif:  OK"
    fi

    mkdir -p models

    echo ""
    echo "=== Setup complete ==="
    echo "Next: sbatch tacc_run.sh train"
    exit 0
fi

# -------------------------------------------------------------------
# TRAIN: run all 3 models sequentially (submitted via SLURM)
# -------------------------------------------------------------------
if [ "$MODE" = "train" ]; then
    echo "=== GRSS Training ==="
    echo "Start time: $(date)"
    echo "Node: $(hostname)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo ""

    # Activate environment
    module load cuda/12.4 2>/dev/null || module load cuda/11.8 2>/dev/null || true
    module load python3/3.11 2>/dev/null || module load python3 2>/dev/null || true
    module load conda 2>/dev/null || true
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
        conda activate grss 2>/dev/null || true
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi

    mkdir -p models results

    # ---- Model 1: PixelMLP ----
    echo "=========================================="
    echo "Training Model 1/2: PixelMLP"
    echo "=========================================="
    python3 train.py --config configs/grss_pixel_mlp.yaml 2>&1 | tee results/grss_pixel_mlp.log
    echo ""

    # ---- Model 2: CASiameseUNet ----
    echo "=========================================="
    echo "Training Model 2/2: CASiameseUNet"
    echo "=========================================="
    python3 train.py --config configs/grss_ca_siamese.yaml 2>&1 | tee results/grss_ca_siamese.log
    echo ""

    echo "=== All training complete ==="
    echo "End time: $(date)"
    echo ""
    echo "Run 'bash tacc_run.sh results' to see summary."
    exit 0
fi

# -------------------------------------------------------------------
# RESULTS: extract mIOU and GDice from logs
# -------------------------------------------------------------------
if [ "$MODE" = "results" ]; then
    echo "=== GRSS Training Results ==="
    echo ""
    printf "%-30s  %-20s  %-10s\n" "Model" "mIOU" "GDice"
    printf "%-30s  %-20s  %-10s\n" "-----" "----" "-----"

    for log in results/grss_*.log; do
        if [ -f "$log" ]; then
            model=$(basename "$log" .log)
            miou=$(grep -oP 'mIOU: \K[^\s,]+' "$log" | tail -1)
            gdice=$(grep -oP 'gdice: \K[^\s]+' "$log" | tail -1)
            printf "%-30s  %-20s  %-10s\n" "$model" "${miou:-N/A}" "${gdice:-N/A}"
        fi
    done

    echo ""
    echo "=== Per-model details ==="
    for log in results/grss_*.log; do
        if [ -f "$log" ]; then
            echo ""
            echo "--- $(basename "$log") ---"
            grep -E "(mIOU|gdice|loss:|saved|Epoch)" "$log" | tail -20
        fi
    done
    exit 0
fi

# -------------------------------------------------------------------
# HELP
# -------------------------------------------------------------------
echo "Usage: bash tacc_run.sh [setup|train|results]"
echo ""
echo "  setup    - Install deps, extract data, verify files (run once)"
echo "  train    - Run all 3 models (submit via: sbatch tacc_run.sh train)"
echo "  results  - Print summary of training results from logs"
