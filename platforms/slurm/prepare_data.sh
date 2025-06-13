#!/bin/bash

#SBATCH --job-name=data-multimodalhugs
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=translation-data.out
#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

# SLURM-specific environment setup
module load cuda anaconda3
source activate ${CONDA_ENV:-multimodalhugs}

# Call the main prepare_data script
make prepare \
  DATA_DIR=${DATA_DIR:-"/scratch/$(whoami)/tmp/signwriting-transcription"} \
  POSES_DIR=${POSES_DIR:-"/shares/sigma.ebling.cl.uzh/$(whoami)/import/poses"} \
  CONFIG_FILE=${CONFIG_FILE:-"signwriting_transcription/config.yaml"} \
  WANDB_PROJECT=${WANDB_PROJECT:-"mmh_transcription"}
