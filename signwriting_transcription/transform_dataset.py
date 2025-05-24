import argparse
import pandas as pd
from pathlib import Path
from signwriting.formats.swu_to_fsw import swu_add_prefix

def get_dataset_tokens(pose: str):
    if pose.startswith("dictio"):
        return ["__dictio__"]

    if pose.startswith("s2m"):
        return ["__sign2mint__"]

    if pose.startswith("ss"):
        return ["__signsuisse__"]

    if pose.startswith("fasl"):
        return ["__fleurs-asl__"]

    return []

def map_datum(row, poses_dir: Path):
    source_tokens = ["__pose__"] + get_dataset_tokens(row['pose'])

    return {
        "signal": str(poses_dir / row['pose']),
        "signal_start": row['start'],
        "signal_end": row['end'],
        "encoder_prompt": " ".join(source_tokens),
        "decoder_prompt": f"__{row['videoLanguage']}__",
        "output": swu_add_prefix(row['text'])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file path")
    parser.add_argument("--poses", type=Path, required=True, help="Directory containing pose files")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV file path")
    args = parser.parse_args()

    # Read input CSV
    print(f"Reading {args.input}")
    df = pd.read_csv(args.input)

    new_tokens = set()

    # Apply mapping function to each row with poses directory
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        split_data = (map_datum(row, args.poses) for _, row in split_df.iterrows())
        split_df = pd.DataFrame(split_data)
        split_file = args.output.with_suffix(f".{split}.tsv")
        print(f"Writing {split_file}")
        split_df.to_csv(split_file, sep='\t', index=False)

        for field in ["encoder_prompt", "decoder_prompt"]:
            for language_tokens in split_df[field]:
                for language_token in language_tokens.split():
                    new_tokens.add(language_token)

    # Write new tokens to file
    tokens_file = args.output.with_suffix(".tokens.txt")
    print(f"Writing {tokens_file}")
    with open(tokens_file, 'w') as f:
        for token in new_tokens:
            f.write(f"{token}\n")

if __name__ == "__main__":
    main()
