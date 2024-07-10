#!/bin/bash
# save the output to a file
exec > >(tee -a pretain_output.log) 2>&1

# Install the required packages
pip install .[dev,pose_to_signwriting]

# Run prepare_pretrain script
python signwriting_transcription/pose_to_signwriting/data/prepare_pretrain.py \
  --data-root pretrain_data_set \
  --dataset-name poses \
  --dataset-size 5000 \

# Run config script
python signwriting_transcription/pose_to_signwriting/data/pretrain_config.py --data-path pretrain_data_set/poses --experiment-dir pretrain_data_set/experiment

# Prepare experiment directory
mkdir -p pretrain_data_set/experiment
cp pretrain_data_set/poses/config.yaml pretrain_data_set/experiment/config.yaml

# Run training script
python signwriting_transcription/pose_to_signwriting/joeynmt_pose/training.py pretrain_data_set/poses/config.yaml
