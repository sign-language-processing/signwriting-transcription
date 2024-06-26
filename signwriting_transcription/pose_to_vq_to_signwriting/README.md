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
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/v2/no-factors" --partition lowprio
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/v2/source-factors" --source-factors --partition lowprio
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/v2/target-factors" --target-factors --partition lowprio # CUDA OOM
sbatch train_sockeye_model.sh "$DATA_DIR/parallel" "$MODEL_DIR/v2/source-target-factors" --source-factors --target-factors --partition lowprio

# 2.1 (Optional) See the validation metrics
watch tail "$MODEL_DIR/no-factors/model/metrics" 

# 3. Evaluate the model
python evaluate.py \
  --hypothesis="$MODEL_DIR/no-factors/test.translations" \
  --reference="$DATA_DIR/parallel/test/target.txt" 
```

## Results



### No-factors

TokenizedBLEU 8.893
CHRF 22.908
SymbolsDistances 27.820
CLIPScore 59.170

v2
TokenizedBLEU 8.632
CHRF 22.478
SymbolsDistances 27.261
CLIPScore 51.128


### Source-factors

TokenizedBLEU 9.720
CHRF 23.604
SymbolsDistances 30.531
CLIPScore 69.483

v2
TokenizedBLEU 9.575
CHRF 23.671
SymbolsDistances 31.632
CLIPScore 69.277


### Target-factors

CUDA OOM https://github.com/awslabs/sockeye/issues/1106

### Source-target-factors

TokenizedBLEU 8.936
CHRF 22.692
SymbolsDistances 27.656
CLIPScore 68.514

v2
TokenizedBLEU 9.262
CHRF 22.996
SymbolsDistances 30.376
CLIPScore 68.880

