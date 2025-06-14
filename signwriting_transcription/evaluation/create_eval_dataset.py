import argparse
import itertools
from pathlib import Path

import pandas as pd

valid_encoder_prompts = [
    # '__pose__ __dictio__',
    '__pose__ __sign2mint__',
    # '__pose__ __signsuisse__',
    # '__pose__ __fleurs-asl__',
]

valid_decoder_prompts = [
    '__ase__',
    '__gsg__',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory with pose files")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV file path")
    args = parser.parse_args()

    # Read input Directory
    pose_files = args.input.glob("*.pose")
    rows = []

    for pose_file in itertools.islice(pose_files, 10):
        for encoder_prompt in valid_encoder_prompts:
            for decoder_prompt in valid_decoder_prompts:
                rows.append({
                    "signal": str(pose_file),
                    "signal_start": 0,
                    "signal_end": 0,
                    "encoder_prompt": encoder_prompt,
                    "decoder_prompt": decoder_prompt,
                    "output": "𝠃𝤘𝥂񌏁𝣴𝣵񍠑𝣿𝤌񀀑𝤄𝤤񈺃𝣜𝤠" # placeholder output
                })

    # save rows as tsv file args.output
    df = pd.DataFrame(rows)
    df.to_csv(args.output, sep='\t', index=False)


if __name__ == "__main__":
    main()

# python -m signwriting_transcription.evaluation.create_eval_dataset --input=/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/test/no --output=popsign_no.tsv
