#!/bin/bash

#SBATCH --job-name=data-multimodalhugs
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=translation-data.out
#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load gpu
module load cuda

module load anaconda3
source activate multimodalhugs

# Download CSV dataset
DATA_DIR="/scratch/$(whoami)/tmp/signwriting-transcription"
mkdir -p $DATA_DIR

[ ! -f "$DATA_DIR/data.csv" ] && \
wget -O "$DATA_DIR/data.csv" https://github.com/sign/data/raw/refs/heads/main/signwriting-transcription/data.csv

# Transform dataset to TSV splits
python "$(dirname "$0")/transform_dataset.py" \
  --input "$DATA_DIR/data.csv" \
  --poses "/shares/sigma.ebling.cl.uzh/amoryo/import/poses" \
  --output "$DATA_DIR/data.tsv"

