import time
import torch
from signwriting.formats.swu_to_fsw import swu2fsw

from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from multimodalhugs.tasks.translation.inference_utils import batched_inference

# Needed for AutoModel to work with the model
import multimodalhugs.models

# Definitions
MODEL_ID = "/scratch/amoryo/tmp/signwriting-transcription/results/signwriting_transcription_model/output/checkpoint-264704"
PROCESSOR_ID = "/scratch/amoryo/tmp/signwriting-transcription/results/pose2text_translation_processor"
TSV_PATH = "/scratch/amoryo/tmp/signwriting-transcription/data.dev.tsv"

# Other
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiation
start_time = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
model_load_time = time.time() - start_time
print(f"Model loading time: {model_load_time:.2f} seconds")

start_time = time.time()
processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
processor_load_time = time.time() - start_time
print(f"Processor loading time: {processor_load_time:.2f} seconds")


start_time = time.time()
output = batched_inference(model=model,
                           processor=processor,
                           tsv_path=TSV_PATH,
                           modality="pose2text",
                           batch_size=64)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

print(f"output['preds']:\n{len(output['preds'])}\n")
with open("output.txt", "w", encoding="utf-8") as f:
    for pred in output['preds']:
        print(swu2fsw(pred))
        f.write(pred + "\n")

