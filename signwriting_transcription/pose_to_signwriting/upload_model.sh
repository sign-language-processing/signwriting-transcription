#!/bin/bash

PRETRAIN=${1:-false}

# Download token.json
wget 'https://drive.google.com/uc?export=download&id=1EwgVIAxa_VcPWMtaFXru19ZBqc8NPq8K' -O signwriting_transcription/pose_to_signwriting/joeynmt_pose/token.json

if [ "$PRETRAIN" != "pretrain" ]; then
  # Modify the config.yaml file to set eval_all_metrics to True
  python signwriting_transcription/pose_to_signwriting/data/config.py --data-path pretrain_data_set/experiment --experiment-dir pretrain_data_set/experiment --test-eval-matrices True

  # Run prediction script
  python signwriting_transcription/pose_to_signwriting/joeynmt_pose/prediction.py pretrain_data_set/experiment/config.yaml test none pretrain_
else
  # Modify the config.yaml file to set eval_all_metrics to True
  python signwriting_transcription/pose_to_signwriting/data/config.py --data-path experiment --experiment-dir experiment --test-eval-matrices True

  # Run prediction script
  python signwriting_transcription/pose_to_signwriting/joeynmt_pose/prediction.py experiment/config.yaml test none
fi
