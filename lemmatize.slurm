#!/bin/bash
#SBATCH --job-name=lemmatize
#SBATCH --output=lemmatize_%j.out
#SBATCH --error=lemmatize_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Paths
PROJECT_DIR="$HOME/topic-modelling-analysis"
IMAGE="$HOME/topic-modelling.sif"

cd "$PROJECT_DIR"

echo "[INFO] Starting lemmatization job"
echo "[INFO] Input: job_test.txt"
echo "[INFO] Output: lemmatized_jobs.txt"

apptainer exec --bind "$PROJECT_DIR":"$PROJECT_DIR" \
    "$IMAGE" python preprocess_lemmatize.py \
        --input job_test.txt \
        --output lemmatized_jobs.txt

echo "[INFO] Lemmatization finished"