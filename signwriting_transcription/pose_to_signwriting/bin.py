#!/usr/bin/env python

import argparse
from pathlib import Path

import os
from tqdm import tqdm
import numpy as np
import pympi
from huggingface_hub import hf_hub_download

from signwriting_transcription.pose_to_signwriting.data.preprocessing import preprocess
from signwriting_transcription.pose_to_signwriting.data.pose_data_utils import build_pose_vocab
from signwriting_transcription.pose_to_signwriting.data.datasets_pose import pose_to_matrix
from signwriting_transcription.pose_to_signwriting.data.config import create_test_config
from signwriting_transcription.pose_to_signwriting.joeynmt_pose.prediction import translate

HUGGINGFACE_REPO_ID = "ohadlanger/signwriting_transcription"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help='model to use')
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--elan', required=True, type=str, help='path to elan file')

    return parser.parse_args()


def main():
    args = get_args()
    print('Downloading model...')
    os.makedirs("experiment", exist_ok=True)
    hf_hub_download(repo_id=HUGGINGFACE_REPO_ID, filename=args.model, output_dir="experiment",
                    file_name="best.ckpt")
    build_pose_vocab(Path('experiment/spm_bpe1182.vocab').absolute())
    create_test_config('experiment', 'experiment')

    print('Loading ELAN file...')
    eaf = pympi.Elan.Eaf(file_path=args.elan, author="sign-language-processing/signwriting-transcription")
    sign_annotations = eaf.get_annotation_data_for_tier('SIGN')

    print('loading sign.....')
    preprocess('.', '.', False)

    print('Predicting signs...')
    temp_files = []
    for index, segment in tqdm(enumerate(sign_annotations)):
        np_pose = pose_to_matrix(args.pose, segment[0], segment[1]).filled(fill_value=0)
        np.save(f'experiment/temp{index}.npy', np_pose)
        temp_files.append(f'experiment/temp{index}.npy')
        eaf.remove_annotation('SIGN', segment[0])
    hyp_list = translate('experiment/config.yaml', temp_files)
    for rm_file in temp_files:
        os.remove(rm_file)
    for index, segment in enumerate(sign_annotations):
        eaf.remove_annotation('SIGN', segment[0])
        eaf.add_annotation('SIGN', segment[0], segment[1], hyp_list[index])
    eaf.to_file(args.elan)


if __name__ == '__main__':
    main()
