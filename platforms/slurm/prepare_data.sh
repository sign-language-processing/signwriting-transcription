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
module load anaconda3
source activate ${CONDA_ENV:-multimodalhugs}

# Call the main prepare_data script
make prepare \
  CONFIG_FILE=${CONFIG_FILE:-"signwriting_transcription/config.yaml"} \
  DATA_DIR=${DATA_DIR:-"/scratch/$(whoami)/tmp/signwriting-transcription-data"} \
  POSES_DIR=${POSES_DIR:-"/shares/sigma.ebling.cl.uzh/$(whoami)/import/poses"} \
  OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/$(whoami)/tmp/signwriting-transcription-fast"}
