#!/bin/bash

#SBATCH --job-name=train-multimodalhugs
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=1
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

# ----------------------------------------------------------
# 1. Specify global variables
# ----------------------------------------------------------
MODEL_NAME="signwriting_transcription_model"
MODEL_DIR="/scratch/amoryo/tmp/signwriting-transcription/results/${MODEL_NAME}"
OUTPUT_PATH="${MODEL_DIR}/output"

MODEL_PATH="${MODEL_DIR}/trained_model"
PROCESSOR_PATH="${MODEL_DIR}/pose2text_translation_processor"
DATA_PATH="${MODEL_DIR}/datasets/pose2text"

# ----------------------------------------------------------
# 2. Train the Model
# ----------------------------------------------------------
# TODO: support signwriting-similarity
multimodalhugs-train \
    --task "translation" \
    --model_name_or_path $MODEL_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --run_name $MODEL_NAME \
    --dataset_dir $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --do_train True \
    --do_eval True \
    --fp16 \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model 'chrf' \
    --overwrite_output_dir \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_steps 20000 \
    --max_steps 200000 \
    --predict_with_generate True \
    --remove_unused_columns False