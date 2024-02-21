#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import numpy as np
import pympi
from tqdm import tqdm

from signwriting_transcription.pose_to_signwriting.data.config import create_test_config
from signwriting_transcription.pose_to_signwriting.data.datasets_pose import pose_to_matrix
from signwriting_transcription.pose_to_signwriting.data.pose_data_utils import build_pose_vocab
from signwriting_transcription.pose_to_signwriting.data.preprocessing import preprocess_single_file
from signwriting_transcription.pose_to_signwriting.joeynmt_pose.prediction import translate

HUGGINGFACE_REPO_ID = "ohadlanger/signwriting_transcription"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--elan', required=True, type=str, help='path to elan file')
    parser.add_argument('--model', type=str, default='bc2de71.ckpt', help='model to use')
    parser.add_argument('--strategies', required=False, type=str, default='tight',
                        choices=['tight', 'wide'], help='strategy to use')
    return parser.parse_args()


def download_model(experiment_dir: Path, model_name: str):
    model_path = experiment_dir / model_name
    if not model_path.exists():
        # pylint: disable=import-outside-toplevel
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=HUGGINGFACE_REPO_ID, filename=model_name, repo_type='space', local_dir='experiment')
        full_path = str(Path('experiment').absolute())
        best_ckpt_path = f'{full_path}/best.ckpt'
        # remove symlink if exists
        if os.path.exists(best_ckpt_path):
            os.remove(best_ckpt_path)
        os.symlink(f'{full_path}/{model_name}', best_ckpt_path)

    vocab_path = experiment_dir / 'spm_bpe1182.vocab'
    if not vocab_path.exists():
        build_pose_vocab(vocab_path.absolute())

    config_path = experiment_dir / 'config.yaml'
    if not config_path.exists():
        create_test_config(str(experiment_dir), str(experiment_dir))


def main():
    args = get_args()

    experiment_dir = Path('experiment')
    experiment_dir.mkdir(exist_ok=True)

    temp_dir = experiment_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)

    print('Downloading model...')
    download_model(experiment_dir, args.model)

    print('Loading ELAN file...')
    eaf = pympi.Elan.Eaf(file_path=args.elan, author="sign-language-processing/signwriting-transcription")
    sign_annotations = eaf.get_annotation_data_for_tier('SIGN')

    print('Preprocessing pose.....')
    temp_pose_path = temp_dir / 'pose.pose'
    preprocess_single_file(args.pose, temp_pose_path, normalization=True)

    print('Predicting signs...')
    temp_files = []
    start = 0
    for index, segment in tqdm(enumerate(sign_annotations)):
        end = sign_annotations[index + 1][0] if index + 1 < len(sign_annotations) else None
        if args.strategies == 'wide':
            end = (end + segment[1]) // 2 if index + 1 < len(sign_annotations) else None
            np_pose = pose_to_matrix(temp_pose_path, start, end).filled(fill_value=0)
            start = end
        else:
            end = end if end is not None else segment[1] + 300
            np_pose = pose_to_matrix(temp_pose_path, segment[0] - (segment[0] - start) * 0.25
                                     , segment[1] + (end - segment[1]) * 0.25).filled(fill_value=0)
            start = segment[1]
        pose_path = temp_dir / f'{index}.npy'
        np.save(pose_path, np_pose)
        temp_files.append(pose_path)

    hyp_list = translate('experiment/config.yaml', temp_files)

    for index, segment in enumerate(sign_annotations):
        eaf.remove_annotation('SIGN', segment[0])
        eaf.add_annotation('SIGN', segment[0], segment[1], hyp_list[index])
    eaf.to_file(args.elan)

    print('Cleaning up...')
    temp_pose_path.unlink()
    for temp_file in temp_files:
        temp_file.unlink()
    temp_dir.rmdir()


if __name__ == '__main__':
    main()
