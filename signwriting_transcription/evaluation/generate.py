import time

import multimodalhugs.models  # noqa: F401  (registers models for HuggingFace AutoModel)
import torch
from multimodalhugs.tasks.translation.inference_utils import batched_inference
from signwriting.formats.swu_to_fsw import swu2fsw
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

# Definitions
MODEL_ID = "/scratch/amoryo/tmp/signwriting-transcription/results/signwriting_transcription_model/output/checkpoint-264704"
PROCESSOR_ID = "/scratch/amoryo/tmp/signwriting-transcription/results/pose2text_translation_processor"
TSV_PATH = "/home/amoryo/sign-language/signwriting-transcription/popsign_no.tsv"

# Other
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiation
start_time = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

start_time = time.time()
processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
print(f"Processor loading time: {time.time() - start_time:.2f} seconds")


start_time = time.time()
output = batched_inference(model=model,
                           processor=processor,
                           tsv_path=TSV_PATH,
                           modality="pose2text",
                           batch_size=64)
print(f"Inference time: {time.time() - start_time:.2f} seconds")

print(f"output['preds']:\n{len(output['preds'])}\n")
with open("output.txt", "w", encoding="utf-8") as f:
    for pred in output['preds']:
        print(swu2fsw(pred))
        f.write(pred + "\n")

