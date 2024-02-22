#!/usr/bin/env python

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pympi
from tqdm import tqdm

from signwriting_transcription.pose_to_signwriting.data.config import create_test_config
from signwriting_transcription.pose_to_signwriting.data.datasets_pose import pose_to_matrix, frame2ms
from signwriting_transcription.pose_to_signwriting.data.pose_data_utils import build_pose_vocab
from signwriting_transcription.pose_to_signwriting.data.preprocessing import preprocess_single_file
from signwriting_transcription.pose_to_signwriting.joeynmt_pose.prediction import translate

HUGGINGFACE_REPO_ID = "ohadlanger/signwriting_transcription"
PADDING_PACTOR = 0.25  # padding factor for tight strategy, 25% padding from both sides of the segment


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--elan', required=True, type=str, help='path to elan file')
    parser.add_argument('--model', type=str, default='bc2de71.ckpt', help='model to use')
    parser.add_argument('--strategies', required=False, type=str, default='tight',
                        choices=['tight', 'wide'], help='segmentation strategy to use')
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

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    print('Downloading model...')
    download_model(experiment_dir, args.model)

    print('Loading ELAN file...')
    eaf = pympi.Elan.Eaf(file_path=args.elan, author="sign-language-processing/signwriting-transcription")
    sign_annotations = eaf.get_annotation_data_for_tier('SIGN')

    print('Preprocessing pose.....')
    preprocessed_pose = preprocess_single_file(args.pose, normalization=False)

    print('Predicting signs...')
    temp_files = []
    start = 0
    # get pose length in ms
    pose_length = frame2ms(len(preprocessed_pose.body.data), preprocessed_pose.body.fps)
    for index, segment in tqdm(enumerate(sign_annotations)):
        if index + 1 < len(sign_annotations):
            end = sign_annotations[index + 1][0]
        else:
            end = pose_length
        if args.strategies == 'wide':   # wide strategy - split the all pose between the segments
            end = (end + segment[1]) // 2
            np_pose = pose_to_matrix(preprocessed_pose, start, end).filled(fill_value=0)
            start = end
        else:   # tight strategy - add padding(PADDING_PACTOR) to the tight segment
            # add padding to the segment by the distance between the segments
            np_pose = pose_to_matrix(preprocessed_pose, segment[0] - (segment[0] - start) * PADDING_PACTOR
                                     , segment[1] + (end - segment[1]) * PADDING_PACTOR).filled(fill_value=0)
            start = segment[1]
        pose_path = temp_path / f'{index}.npy'
        np.save(pose_path, np_pose)
        temp_files.append(pose_path)

    hyp_list = translate('experiment/config.yaml', temp_files)

    for index, segment in enumerate(sign_annotations):
        eaf.remove_annotation('SIGN', segment[0])
        eaf.add_annotation('SIGN', segment[0], segment[1], hyp_list[index])
    eaf.to_file(args.elan)

    print('Cleaning up...')
    for temp_file in temp_files:
        temp_file.unlink()
    temp_path.rmdir()


if __name__ == '__main__':
    main()
