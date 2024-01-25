#!/bin/bash

#SBATCH --job-name=train-sockeye
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --output=train.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB|GPUMEM32GB

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate sockeye

mkdir -p $2

# Clone sockeye if doesn't exist
[ ! -d sockeye ] && git clone https://github.com/awslabs/sockeye.git
cd sockeye

pip install -r requirements/requirements.txt


function find_source_files() {
    local directory=$1
    find "$directory" -type f -name 'source_[1-9]*.txt' -printf "$directory/%f\n" | sort | tr '\n' ' '
}

function find_target_files() {
    local directory=$1
    find "$directory" -type f -name 'target_[1-9]*.txt' -printf "$directory/%f "
}

# Prepare data
TRAIN_DATA_DIR="$2/factored_train_data"
[ ! -d "$TRAIN_DATA_DIR" ] && \
python -m sockeye.prepare_data \
  --source $1/train/source_0.txt --source-factors $(find_source_files "$1/train") \
  --target $1/train/target_0.txt --target-factors $(find_target_files "$1/train")   \
  --output $TRAIN_DATA_DIR


MODEL_DIR="$2/model"
rm -rf $MODEL_DIR

# batch size refers to number of target tokens
# TODO change optimized-metric
python -m sockeye.train \
  -d $TRAIN_DATA_DIR \
  --weight-tying-type none \
  --batch-size 128 \
  --source-factors-combine sum \
  --target-factors-combine sum \
  --validation-source $1/dev/source_0.txt --validation-source-factors $(find_source_files "$1/dev") \
  --validation-target $1/dev/target_0.txt --validation-target-factors $(find_target_files "$1/dev") \
  --optimized-metric chrf \
  --decode-and-evaluate 500 \
  --checkpoint-interval 500 \
  --max-num-checkpoint-not-improved 20 \
  --output $MODEL_DIR

# Run predictions on test set
python -m sockeye.translate \
  -m $MODEL_DIR \
  --input $1/test/source_0.txt --input-factors $(find_source_files "$1/test") \
  --output $2/test.translations.factors \
  --output-type translation_with_factors

# Replace "|" with space, replace "M c0 r0" with "M"
cat $2/test.translations.factors | sed 's/|/ /g' | sed 's/M c0 r0/M/g' > $2/test.translations

