#!/bin/bash

# Clone the repository
git clone https://github.com/sign-language-processing/signwriting-transcription.git
cd signwriting-transcription

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

# Download token.json
wget 'https://drive.google.com/uc?export=download&id=1EwgVIAxa_VcPWMtaFXru19ZBqc8NPq8K' -O signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json

# Modify the config.yaml file to set eval_all_metrics to True
python signwriting_transcription/pose_to_signwriting/data/pretrain_config.py --data-path pretrain_data_set/experiment --experiment-dir pretrain_data_set/experiment --test-eval-matrices True

# Run prediction script
python signwriting_transcription/pose_to_signwriting/joeynmt_pose/prediction.py experiment/config.yaml test none pretrain_
