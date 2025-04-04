#!/bin/bash

#SBATCH --job-name=train-multimodalhugs
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=translation-job.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

module load gpu
module load cuda

module load anaconda3
source activate multimodalhugs

# Login to huggingface
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Verify GPU with PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# Set WANDB project
export WANDB_PROJECT=mmh_transcription

# Specify global variables
MODEL_NAME="signwriting_transcription_model"
MODEL_DIR="/scratch/amoryo/tmp/signwriting-transcription/results/${MODEL_NAME}"
OUTPUT_PATH="${MODEL_DIR}/output"

# Run setup (again) to ensure the config file is up-to-date with paths for the processor etc
multimodalhugs-setup \
  --modality "pose2text" \
  --config-path "signwriting_transcription/config.yaml"

# Train the Model
# TODO: support signwriting-similarity metric
multimodalhugs-train \
  --task "translation" \
  --config-path "signwriting_transcription/config.yaml" \
  --output_dir "$OUTPUT_PATH"