#!/bin/bash
#SBATCH --job-name=topic_model
#SBATCH --output=topic_model_%j.out
#SBATCH --error=topic_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL

# Paths
PROJECT_DIR="$HOME/topic-modelling-analysis"
IMAGE="$HOME/topic-modelling.sif"

cd "$PROJECT_DIR"

if [ -z "$1" ]; then
    echo "[ERROR] No input file provided!"
    echo "Usage: sbatch topic_model.slurm job_test.txt"
    exit 1
fi

echo "[INFO] Starting topic modelling"
echo "[INFO] Input: $INPUT_FILE"

apptainer exec --bind "$PROJECT_DIR":"$PROJECT_DIR" \
    "$IMAGE" python topic_modelling.py \
        --input "$INPUT_FILE" \
        --lemmatized

echo "[INFO] Topic modelling finished"