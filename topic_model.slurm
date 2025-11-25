#!/bin/bash
#SBATCH --job-name=topic_model
#SBATCH --output=topic_model_%j.out
#SBATCH --error=topic_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Paths
PROJECT_DIR="$HOME/topic-modelling-analysis"
IMAGE="$HOME/topic-modelling.sif"

cd "$PROJECT_DIR"

echo "[INFO] Starting topic modelling"
echo "[INFO] Input: lemmatized_jobs.txt"

apptainer exec --bind "$PROJECT_DIR":"$PROJECT_DIR" \
    "$IMAGE" python topic_modelling.py \
        --input lemmatized_jobs.txt \
        --lemmatized

echo "[INFO] Topic modelling finished"