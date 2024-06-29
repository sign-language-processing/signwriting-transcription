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
from signwriting_transcription.pose_to_signwriting.data.pose_data_utils import (
    build_sp_model,
    create_zip,
    get_zip_manifest,
    save_tsv,
    build_pose_vocab
)
from signwriting_transcription.pose_to_signwriting.data.datasets_pose import (
    load_dataset, extract_to_matrix, frame2ms, pose_ndarray_to_matrix
)

COLUMNS = ["id", "src", "n_frames", "trg"]

SEED = 123
N_MEL_FILTERS = 534
N_WORKERS = 4  # cpu_count()
SP_MODEL_TYPE = "bpe"  # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 1182  # joint vocab
EXPANDED_DATASET = 1000  # the minimum number of samples in the dataset


def get_split_data(dataset, feature_root):
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
            "trg": instance[2],
            "split": instance[3]
        })
    return all_data


def process(args):
    # pylint: disable=too-many-locals
    data_root, name, size = (
        args.data_root, args.dataset_name, int(args.dataset_size))
    cur_root = Path(data_root).absolute()
    cur_root = cur_root / name

    # dir for filterbank (shared across splits)
    feature_root = cur_root / f"fbank{N_MEL_FILTERS}"
    feature_root.mkdir(parents=True, exist_ok=True)
    const_np_array = np.zeros((10, 534))
    const_np_array[0][0] = -9999

    dataset = []
    for index in range(size):
        instance = [str(index), const_np_array.copy(), "SYNTHETIC", "train"]
        dataset.append(tuple(instance))
    for index in range(30):
        test_instance = [f"test{index}", const_np_array.copy(), "SYNTHETIC", "test"]
        dev_instance = [f"dev{index}", const_np_array.copy(), "SYNTHETIC", "dev"]
        dataset.append(tuple(test_instance))
        dataset.append(tuple(dev_instance))
    print("the length of dataset: ", len(dataset))

    print("Extracting pose features ...")
    for instance in dataset:
        utt_id = instance[0]
        extract_to_matrix(instance[1], feature_root / f'{utt_id}.npy', overwrite=False)

    # Pack features into ZIP
    print("ZIPing features...")
    create_zip(feature_root, feature_root.with_suffix(".zip"))

    all_data = get_split_data(dataset, feature_root)

    all_df = pd.DataFrame.from_records(all_data)
    save_tsv(all_df, cur_root / "poses_all_data.tsv")

    for split in ['train', 'dev', 'test']:
        split_df = all_df[all_df['split'] == split]
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
    build_pose_vocab(cur_root / f"spm_bpe{VOCAB_SIZE}.vocab")
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--dataset-size", required=False, type=str, default=True)
    args = parser.parse_args()
    # alert if the size is smaller then the expected size
    assert int(args.dataset_size) >= EXPANDED_DATASET
    process(args)


if __name__ == "__main__":
    main()
