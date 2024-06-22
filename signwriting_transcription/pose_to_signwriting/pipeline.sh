#!/bin/bash

# Clone the repository
git clone https://github.com/sign-language-processing/signwriting-transcription.git
cd signwriting-transcription

# Install the required packages
pip install .[dev,pose_to_signwriting]

# Download and unzip the transcription data set
wget -O transcription.zip "https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Ftranscription.zip?alt=media"
unzip transcription.zip -d transcription_data_set

# Run preprocessing script
python signwriting_transcription/pose_to_signwriting/data/preprocessing.py --src-dir transcription_data_set --trg-dir normalized_data_set --normalization True

# Prepare segmentation data set
mkdir -p segment_data_set
cp data/data_segmentation.csv segment_data_set/target.csv
cp data/data.csv normalized_data_set/target.csv

# Run prepare_poses script
python signwriting_transcription/pose_to_signwriting/data/prepare_poses.py \
  --dataset-root normalized_data_set \
  --data-root vectorized_data_set \
  --dataset-name poses \
  --tokenizer-type pose-vpf \
  --data-segment segment_data_set

# Run config script
python signwriting_transcription/pose_to_signwriting/data/config.py --data-path vectorized_data_set/poses --experiment-dir experiment

# Prepare experiment directory
mkdir -p experiment
cp vectorized_data_set/poses/config.yaml experiment/config.yaml

# Run training script
python signwriting_transcription/pose_to_signwriting/joeynmt_pose/training.py vectorized_data_set/poses/config.yaml

# Download token.json
wget 'https://drive.google.com/uc?export=download&id=1EwgVIAxa_VcPWMtaFXru19ZBqc8NPq8K' -O signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json

# Modify the config.yaml file to set eval_all_metrics to True
python signwriting_transcription/pose_to_signwriting/data/config.py --data-path experiment --experiment-dir experiment --test-eval-matrices True

# Run prediction script
python signwriting_transcription/pose_to_signwriting/joeynmt_pose/prediction.py experiment/config.yaml test none
