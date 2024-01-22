#!/usr/bin/env python
# coding: utf-8
"""
Prepare poses

expected dir structure:
    vectorized_data_set/
    └── poses/
            ├── fbank534/
            │   ├── test1.npy
            │   ├── test2.npy
            │   ├── test3.npy
            ├── fbank534.zip
            ├── joey_train_asr.tsv
            ├── joey_dev_asr.tsv
            └── joey_test_asr.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from joeynmt.helpers import write_list_to_file
from pose_data_utils import (
    build_sp_model,
    create_zip,
    get_zip_manifest,
    save_tsv,
    build_pose_vocab
)
from datasets_pose import load_dataset, extract_to_matrix

COLUMNS = ["id", "src", "n_frames", "trg"]

SEED = 123
N_MEL_FILTERS = 534
N_WORKERS = 4  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 1182  # joint vocab
EXPANDED_DATASET = 1000  # the minimum number of samples in the dataset


def process(args, pumping: bool = False):
    # pylint: disable=too-many-locals
    dataset_root, data_root, name, tokenizer_type = (
        args.dataset_root, args.data_root, args.dataset_name, args.tokenizer_type,)
    cur_root = Path(data_root).absolute()
    cur_root = cur_root / name

    # dir for filterbank (shared across splits)
    feature_root = cur_root / f"fbank{N_MEL_FILTERS}"
    feature_root.mkdir(parents=True, exist_ok=True)

    # Extract features
    print(f"Create pose {name} dataset.")

    print("Fetching train split ...")
    dataset = load_dataset(dataset_root)

    print("Extracting pose features ...")
    for instance in dataset:
        utt_id = instance[0]
        extract_to_matrix(instance[1], feature_root / f'{utt_id}.npy', overwrite=False)

    # Pack features into ZIP
    print("ZIPing features...")
    create_zip(feature_root, feature_root.with_suffix(".zip"))

    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))

    # Generate TSV manifest
    print("Generating manifest...")
    all_data = []

    for instance in dataset:
        utt_id = instance[0]
        n_frames = np.load(feature_root / f'{utt_id}.npy').shape[0]
        all_data.append({
            "id": utt_id,
            "src": zip_manifest[str(utt_id)],
            "n_frames": n_frames,
            "trg": instance[2]
        })

    if EXPANDED_DATASET > len(all_data) and pumping:
        print("Pumping dataset...")
        backup = all_data.copy()
        for i in range(EXPANDED_DATASET - len(backup)):
            utt_id = backup[i % len(backup)]["id"]
            n_frames = backup[i % len(backup)]["n_frames"]
            trg = backup[i % len(backup)]["trg"]
            src = backup[i % len(backup)]["src"]
            all_data.append({
                "id": f'{utt_id}({i})',  # unique id
                "src": src,
                "n_frames": n_frames,
                "trg": trg
            })

    all_df = pd.DataFrame.from_records(all_data)
    save_tsv(all_df, cur_root / "poses_all_data.tsv")

    # Split the data into train and test set and save the splits in tsv
    np.random.seed(SEED)
    probs = np.random.rand(len(all_df))
    mask = {}
    dev_range = np.partition(probs, 300 - 1)[300 - 1]
    test_range = np.partition(probs, 150 - 1)[150 - 1]
    mask['train'] = probs > dev_range
    mask['dev'] = (probs <= dev_range) & (probs > test_range)
    mask['test'] = probs <= test_range

    for split in ['train', 'dev', 'test']:
        split_df = all_df[mask[split]]
        # save tsv
        save_tsv(split_df, cur_root / f"{split}.tsv")
        # save plain txt
        write_list_to_file(cur_root / f"{split}.txt", split_df['trg'].to_list())
        print(split, len(split_df))

    # Generate joint vocab
    print("Building joint vocab...")
    kwargs = {
        'model_type': SP_MODEL_TYPE,
        'vocab_size': VOCAB_SIZE,
        'character_coverage': 1.0,
        'num_workers': N_WORKERS
    }
    if tokenizer_type == 'pose-vpf':
        build_pose_vocab(cur_root / f"spm_bpe{VOCAB_SIZE}.vocab")
    else:
        build_sp_model(cur_root / "train.txt", cur_root / f"spm_bpe{VOCAB_SIZE}", **kwargs)
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--tokenizer-type", required=True, type=str)
    args = parser.parse_args()
    process(args, pumping=True)


if __name__ == "__main__":
    main()
