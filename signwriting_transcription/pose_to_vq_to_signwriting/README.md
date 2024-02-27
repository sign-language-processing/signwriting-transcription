# Pose-to-VQ-to-SignWriting

Using the [sign-vq](https://github.com/sign-language-processing/sign-vq) model, we can compress a pose sequence
into a sequence of discrete tokens.
Then, the translation into SignWriting is done using a text-to-text machine translation model.

## Factored Machine Translation

While not mandatory, we use a factored machine translation model.

### Source Factors

Given that our VQ model returns 4 tokens per frame, and we do not want to increase our sequence length by 4x,
we factorize the source sequence into 4 factors, each embedded and then summed to form the final source embedding.

### Target Factors

SignWriting signs are predictable - a box must have a position, and every symbol must have two modifiers and a position.
Therefore, it fits nicely into a factorized representation, which decreases the sequence length by 5x,
and enforces correctly formatted output generation.

Our factors are: Base symbol, modifier1, modifier2, x position, y position

## TODOs:

- [ ] Retrain VQ model to perform better on hand shapes
- [ ] Add --source-factors and --target-factors to `train_sockeye_model.sh` as options
- [ ] Train and compare the following models:
  - [ ] A model with no factors
  - [ ] A model with source factors but no target factors
  - [ ] A model with source and target factors
  - [ ] A model with target factors but no source factors

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
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/no-factors"
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/source-factors" --source-factors
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/target-factors" --target-factors
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/source-target-factors" --source-factors --target-factors

# 2.1 (Optional) See the validation metrics
watch tail "$MODEL_DIR/no-factors/model/metrics" 

# 3. Evaluate the model
python evaluate.py \
  --hypothesis="$MODEL_DIR/no-factors/test.translations" \
  --reference="$DATA_DIR/parallel/test/target.txt" 
  
#TokenizedBLEU 5.776
#CHRF 22.123
#SymbolsDistances 22.420
#CLIPScore 85.616
```
