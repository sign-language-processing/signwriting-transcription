#!/bin/bash

#SBATCH --job-name=train-sockeye
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --output=train-%j.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate sockeye

mkdir -p $2

# Clone sockeye if doesn't exist
#[ ! -d sockeye ] && git clone https://github.com/sign-language-processing/sockeye.git
#pip install ./sockeye
#
## Install SignWriting evaluation package for optimized metric
#pip install git+https://github.com/sign-language-processing/signwriting
#pip install git+https://github.com/sign-language-processing/signwriting-evaluation
#pip install tensorboard

function find_source_files() {
    local directory=$1
    find "$directory" -type f -name 'source_[1-9]*.txt' -printf "$directory/%f\n" | sort | tr '\n' ' '
}

function find_target_files() {
    local directory=$1
    find "$directory" -type f -name 'target_[1-9]*.txt' -printf "$directory/%f "
}

# Flags for source and target factors
use_source_factors=false
use_target_factors=false

# Check command line arguments for --source-factors and --target-factors
for arg in "$@"
do
    if [[ "$arg" == "--source-factors" ]]; then
        use_source_factors=true
    elif [[ "$arg" == "--target-factors" ]]; then
        use_target_factors=true
    fi
done

function translation_files() {
    local name=$1
    local type=$2    # e.g., "source" or "target"
    local split=$3   # e.g., "train", "dev", or "test"
    local use_factors=$4  # Pass 'true' or 'false' to use factors

    # Determine the file finder function based on the type
    local find_function="find_${type}_files"

    if [[ "$use_factors" == "true" ]]; then
        echo "--${name} ${split}/${type}_0.txt --${name}-factors $($find_function "$split")"
    else
        echo "--${name} ${split}/${type}.txt"
    fi
}

# Prepare data
TRAIN_DATA_DIR="$2/train_data"
[ ! -f "$TRAIN_DATA_DIR/data.version" ] && \
python -m sockeye.prepare_data \
  --max-seq-len 1024 \
  $(translation_files "source" "source" "$1/train" $use_source_factors) \
  $(translation_files "target" "target" "$1/train" $use_target_factors) \
  --output $TRAIN_DATA_DIR


MODEL_DIR="$2/model"
rm -rf $MODEL_DIR

# batch size refers to number of target tokens, has to be larger than max tokens set in prepare_data
python -m sockeye.train \
  -d $TRAIN_DATA_DIR \
  --weight-tying-type none \
  --batch-size 1028 \
  --num-layers 4:4 \
  --source-factors-combine sum \
  --target-factors-combine sum \
  $(translation_files "validation-source" "source" "$1/dev" $use_source_factors) \
  $(translation_files "validation-target" "target" "$1/dev" $use_target_factors) \
  --optimized-metric signwriting-similarity \
  --decode-and-evaluate 500 \
  --checkpoint-interval 500 \
  --max-num-checkpoint-not-improved 20 \
  --output $MODEL_DIR


# Run predictions on test set
python -m sockeye.translate \
  -m $MODEL_DIR \
  $(translation_files "input" "source" "$1/test" $use_source_factors) \
  --output $2/test.translations.factors \
  --output-type translation_with_factors

# Replace "|" with space, replace "M c0 r0" with "M"
cat $2/test.translations.factors | sed 's/|/ /g' | sed 's/M c0 r0/M/g' > $2/test.translations

# Evaluate test predictions
python evaluate.py \
  --hypothesis="$2/test.translations" \
  --reference="$1/test/target.txt" > $2/evaluation.txt
