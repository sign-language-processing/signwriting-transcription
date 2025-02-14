#!/bin/bash

PRETRAIN=${1:-false}

# Download token.json
wget 'https://drive.google.com/uc?export=download&id=1W_FS2CHdpoFWezh48-S2DevKFk_mSmzR' -O signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json

if [ "$PRETRAIN" == "pretrain" ]; then
  # Modify the config.yaml file to set eval_all_metrics to True
  python signwriting_transcription/pose_to_signwriting/data/pretrain_config.py --data-path pretrain_data_set/poses --experiment-dir pretrain_data_set/experiment --test-eval-matrices True --model pretrain_data_set/experiment

  # change the location of the file
  cp pretrain_data_set/poses/config.yaml pretrain_data_set/experiment/config.yaml

  # Run prediction script
  python signwriting_transcription/pose_to_signwriting/joeynmt_pose/prediction.py pretrain_data_set/experiment/config.yaml test --add_to_name pretrain_ --ckpt_dir pretrain_data_set/experiment
else
  # Modify the config.yaml file to set eval_all_metrics to True
  python signwriting_transcription/pose_to_signwriting/data/config.py --data-path vectorized_data_set/poses --experiment-dir experiment --test-eval-matrices True --model experiment/best.ckpt

  # change the location of the file
  cp vectorized_data_set/poses/config.yaml experiment/config.yaml

  # Run prediction script
  python signwriting_transcription/pose_to_signwriting/joeynmt_pose/prediction.py experiment/config.yaml test --ckpt_dir experiment
fi
