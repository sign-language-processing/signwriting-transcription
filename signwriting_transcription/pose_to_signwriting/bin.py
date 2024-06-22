#!/usr/bin/env python

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pympi
from pose_format import Pose
from tqdm import tqdm

from signwriting_transcription.pose_to_signwriting.data.config import create_test_config
from signwriting_transcription.pose_to_signwriting.data.datasets_pose import (pose_to_matrix, frame2ms,
                                                                              pose_ndarray_to_matrix)
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
    parser.add_argument('--strategy', type=str, default='tight',
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


def preprocessing_signs(preprocessed_pose: Pose, sign_annotations: list, strategy: str, temp_dir: str):
    temp_files = []  # list of temporary files
    start_point = 0
    temp_path = Path(temp_dir)
    # get pose length in ms
    pose_length = frame2ms(len(preprocessed_pose.body.data), preprocessed_pose.body.fps)
    for index, (sign_start, sign_end, _) in tqdm(enumerate(sign_annotations)):
        if index + 1 < len(sign_annotations):
            end_point = sign_annotations[index + 1][0]
        else:
            end_point = pose_length
        if strategy == 'wide':  # wide strategy - split the all pose between the segments
            end_point = (end_point + sign_start) // 2
            np_pose, frame_rate = pose_to_matrix(preprocessed_pose)
            np_pose = pose_ndarray_to_matrix(np_pose, start_point, frame_rate, end_point).filled(fill_value=0)
            start_point = end_point
        else:  # tight strategy - add padding(PADDING_PACTOR) to the tight segment
            # add padding to the segment by the distance between the segments
            np_pose, frame_rate = pose_to_matrix(preprocessed_pose)
            np_pose = pose_ndarray_to_matrix(np_pose, sign_start - (sign_start - start_point) * PADDING_PACTOR, frame_rate,
                                   sign_end + (end_point - sign_end) * PADDING_PACTOR).filled(fill_value=0)
            start_point = sign_end
        pose_path = temp_path / f'{index}.npy'
        np.save(pose_path, np_pose)
        temp_files.append(pose_path)
    return temp_files


def main():
    args = get_args()

    experiment_dir = Path('experiment')
    experiment_dir.mkdir(exist_ok=True)

    print('Downloading model...')
    download_model(experiment_dir, args.model)

    print('Loading ELAN file...')
    eaf = pympi.Elan.Eaf(file_path=args.elan, author="sign-language-processing/signwriting-transcription")
    sign_annotations = eaf.get_annotation_data_for_tier('SIGN')

    print('Preprocessing pose.....')
    preprocessed_pose = preprocess_single_file(args.pose, normalization=False)

    print('Predicting signs...')
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = preprocessing_signs(preprocessed_pose, sign_annotations, args.strategy, temp_dir)
        hyp_list = translate('experiment/config.yaml', temp_files)

    for index, (start, end, _) in enumerate(sign_annotations):
        eaf.remove_annotation('SIGN', start)
        eaf.add_annotation('SIGN', start, end, hyp_list[index])
    eaf.to_file(args.elan)


if __name__ == '__main__':
    main()
