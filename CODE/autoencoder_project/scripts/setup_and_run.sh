#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- [1/6] STARTING SETUP AND RUN SCRIPT ---"

# --- Variables ---
# [ACCEPTED] Using $HOME for portability instead of a hardcoded path.
PROJECT_DIR="$HOME/autoencoder_project"
# Replace with your actual GCS bucket name
GCS_BUCKET_NAME="srgan-bucket-ace-botany-453819-t4"
INPUT_CSV_GCS_PATH="gs://${GCS_BUCKET_NAME}/data/property_features_embeddings_v4_final.csv"

# --- Setup ---
echo "--- [2/6] UPDATING SYSTEM AND INSTALLING PYTHON ---"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

echo "--- [3/6] CREATING PROJECT DIRECTORY AND VIRTUAL ENVIRONMENT ---"
# This script assumes the project has been uploaded via SCP to $HOME/autoencoder_project
cd "${PROJECT_DIR}"
python3 -m venv venv
source venv/bin/activate

echo "--- [4/6] INSTALLING PYTHON DEPENDENCIES ---"
pip install --upgrade pip
# The requirements file is expected to be in the 'src' subdirectory
pip install -r src/requirements.txt

echo "--- [5/6] DOWNLOADING INPUT DATA FROM GCS ---"
mkdir -p data
gsutil cp "${INPUT_CSV_GCS_PATH}" data/

echo "--- [6/6] RUNNING THE PYTHON TRAINING SCRIPT ---"
# The command is defined as a variable for clarity
TRAIN_COMMAND="python3 src/train_autoencoders.py \
    --input-csv \"data/property_features_embeddings_v4_final.csv\" \
    --output-csv \"output/final_model_input.csv\" \
    --model-dir \"output/models\" \
    --epochs 75 \
    --batch-size 256 \
    --n-trials 100" # <--- ADD THIS NEW LINE

# Execute the command
eval $TRAIN_COMMAND

echo "--- SCRIPT FINISHED SUCCESSFULLY! ---"
echo "You can now download results from the 'output' directory using 'gcloud compute scp'."