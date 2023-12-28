#!/usr/bin/env python3
# coding: utf-8

# Adapted from
# https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/data_utils.py

import csv
import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sentencepiece as sp
from tqdm import tqdm

from joeynmt.constants import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)
from joeynmt.helpers_for_pose import _is_npy_data


def get_zip_manifest(zip_path: Path, npy_root: Optional[Path] = None):
    manifest = {}
    with zipfile.ZipFile(zip_path, mode="r") as f:
        info = f.infolist()
    # retrieve offsets
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        with zip_path.open("rb") as f:
            f.seek(offset)
            data = f.read(file_size)
            assert len(data) > 1 and _is_npy_data(data), (utt_id, len(data))
        manifest[utt_id] = f"{zip_path.name}:{offset}:{file_size}"
        # sanity check
        if npy_root is not None:
            byte_data = np.load(io.BytesIO(data))
            npy_data = np.load((npy_root / f"{utt_id}.npy").as_posix())
            assert np.allclose(byte_data, npy_data)
    return manifest


def create_zip(data_root: Path, zip_path: Path):
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            try:
                f.write(path, arcname=path.name)
            except Exception as e:  # pylint: disable=broad-except
                raise Exception(f"{path}") from e


def save_tsv(df: pd.DataFrame, path: Path, header: bool = True) -> None:
    df.to_csv(path.as_posix(),
              sep="\t",
              header=header,
              index=False,
              encoding="utf-8",
              escapechar="\\",
              quoting=csv.QUOTE_NONE)


def load_tsv(path: Path):
    return pd.read_csv(path.as_posix(),
                       sep="\t",
                       header=0,
                       encoding="utf-8",
                       escapechar="\\",
                       quoting=csv.QUOTE_NONE,
                       na_filter=False)


def build_sp_model(input_path: Path, model_path_prefix: Path, **kwargs):
    """
    Build sentencepiece model
    """
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={model_path_prefix.as_posix()}",
        f"--model_type={kwargs.get('model_type', 'unigram')}",
        f"--vocab_size={kwargs.get('vocab_size', 5000)}",
        f"--character_coverage={kwargs.get('character_coverage', 1.0)}",
        f"--num_threads={kwargs.get('num_workers', 1)}",
        f"--unk_piece={UNK_TOKEN}",
        f"--bos_piece={BOS_TOKEN}",
        f"--eos_piece={EOS_TOKEN}",
        f"--pad_piece={PAD_TOKEN}",
        f"--unk_id={UNK_ID}",
        f"--bos_id={BOS_ID}",
        f"--eos_id={EOS_ID}",
        f"--pad_id={PAD_ID}",
        "--vocabulary_output_piece_score=false",
    ]
    if 'user_defined_symbols' in kwargs:
        arguments.append(f"--user_defined_symbols={kwargs['user_defined_symbols']}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))