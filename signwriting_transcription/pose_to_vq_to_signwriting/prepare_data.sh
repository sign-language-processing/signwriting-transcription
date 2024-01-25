#!/bin/bash

#SBATCH --job-name=prepare-data
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=prepare_data.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq-transcription


mkdir -p $1
ZIP_PATH="$1/dataset.zip"

# Downloads transcription dataset
[ ! -f "$ZIP_PATH" ] && \
wget -O "$ZIP_PATH" "https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Ftranscription.zip?alt=media"

# Install SignWriting package
pip install git+https://github.com/sign-language-processing/signwriting.git

# Install Quantization package
pip install git+https://github.com/sign-language-processing/sign-vq.git

# Quantize the dataset
QUANTIZED_PATH="$1/quantized.csv"
[ ! -f "$QUANTIZED_PATH" ] && \
poses_to_codes \
  --data="$ZIP_PATH" \
  --output="$QUANTIZED_PATH"

# Prepare the parallel corpus (with source/target-factors)
python create_parallel_data.py \
  --codes="$QUANTIZED_PATH" \
  --data="../../data/data.csv" \
  --output-dir="$1/parallel"