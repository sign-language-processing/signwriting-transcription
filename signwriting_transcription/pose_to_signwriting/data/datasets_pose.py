import csv
import re
from typing import Union

import numpy as np
from pose_format import Pose
from signwriting.formats.swu_to_fsw import swu2fsw


def fsw_cut(fsw_text: str) -> str:
    match = re.search('[MRLB]', fsw_text)
    if match is not None:
        return fsw_text[match.start():]
    return fsw_text


def ms2frame(ms, frame_rate) -> int:
    ms = int(ms)
    return int(ms / 1000 * frame_rate)


def frame2ms(frame, frame_rate) -> int:
    return int(frame * 1000 / frame_rate)


def pose_ndarray_to_matrix(pose_data: np.ndarray, start_ms, frame_rate, end_ms=None):
    start_frame = ms2frame(start_ms, frame_rate)
    end_frame = ms2frame(end_ms, frame_rate) if end_ms is not None else None
    pose_data = pose_data[start_frame:end_frame]
    return pose_data


def pose_to_matrix(file_path_or_pose: Union[str, Pose]):
    if isinstance(file_path_or_pose, str):
        with open(file_path_or_pose, "rb") as file:
            pose = Pose.read(file.read())
    else:
        pose = file_path_or_pose
    frame_rate = 29.97003 if file_path_or_pose == '19097be0e2094c4aa6b2fdc208c8231e.pose' else pose.body.fps
    pose = pose.body.data
    pose = pose.reshape(len(pose), -1)
    return pose, frame_rate


def load_dataset(target_folder, data_folder):
    with open(f'{target_folder}/target.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        dataset = []
        for line in reader:
            try:
                pose, fps = pose_to_matrix(f"{data_folder}/{line['pose']}")
            except FileNotFoundError:
                continue
            pose = pose.filled(fill_value=0)
            utt_id = line['pose'].split('.')[0]
            utt_id = f"{utt_id},{line['start']},{line['end']},{fps}"
            dataset.append((utt_id, pose, fsw_cut(swu2fsw(line['text'])), line['split']))
    return dataset


def extract_to_matrix(pose_data, output_path, overwrite: bool = False):
    if output_path is not None and output_path.is_file() and not overwrite:
        return np.load(output_path.as_posix())
    if output_path is not None:
        np.save(output_path.as_posix(), pose_data)
        assert output_path.is_file(), output_path
    return pose_data
