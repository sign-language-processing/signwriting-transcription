#!/bin/bash

#SBATCH --job-name=train-multimodalhugs
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=translation-job.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

# SLURM-specific environment setup
module load gpu cuda anaconda3
source activate ${CONDA_ENV:-multimodalhugs}

# Call the main train script
make train \
  DATA_DIR=${DATA_DIR:-"/scratch/$(whoami)/tmp/signwriting-transcription-data"} \
  POSES_DIR=${POSES_DIR:-"/shares/sigma.ebling.cl.uzh/$(whoami)/import/poses"} \
  OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/$(whoami)/tmp/signwriting-transcription-fast"} \
  WANDB_PROJECT=${WANDB_PROJECT:-"mmh_transcription"} \
  CONFIG_FILE=${CONFIG_FILE:-"signwriting_transcription/config.yaml"}