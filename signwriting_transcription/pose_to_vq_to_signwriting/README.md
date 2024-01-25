# Pose-to-VQ-to-SignWriting

Using the [sign-vq](https://github.com/sign-language-processing/sign-vq) model, we can compress a pose sequence
into a sequence of discrete tokens.
Then, the translation into SignWriting is done using a text-to-text machine translation model.

## Steps

```bash
# 0. Setup the environment.
conda create --name vq-transcription python=3.11 -y
conda activate vq-transcription

cd signwriting_transcription/pose_to_vq_to_signwriting

# 1. Download and quantize transcription dataset
DATA_DIR=/scratch/amoryo/transcription
sbatch prepare_data.sh "$DATA_DIR"

# 2. Trains a translation model
MODEL_DIR=/shares/volk.cl.uzh/amoryo/checkpoints/sockeye-vq
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR"

# 2.1 (Optional) See the validation metrics
watch tail "$MODEL_DIR/model/metrics" 
```
