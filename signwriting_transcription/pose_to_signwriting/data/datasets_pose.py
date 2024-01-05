import csv
import re
import numpy as np
from pose_format import Pose
from signwriting.formats.swu_to_fsw import swu2fsw

FRAME_RATE = 29.97003


def fsw_cut(fswText: str) -> str:
    match = re.search('[MRLB]', fswText)
    if match is not None:
        return fswText[match.start():]
    else:
        return fswText


def ms2frame(ms) -> int:
    ms = int(ms)
    return int(ms / 1000 * FRAME_RATE)


def pose_to_matrix(file_path, start_ms, end_ms):
    with open(file_path, "rb") as f:
        pose = Pose.read(f.read())
    pose = pose.body.data
    pose = pose.reshape(len(pose), -1)
    # TODO: use pose.body.fps except for one exception
    pose = pose[ms2frame(start_ms):ms2frame(end_ms)]
    return pose


def load_dataset(folder_name):
    with open(f'{folder_name}/target.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        dataset = []
        for line in reader:
            pose = pose_to_matrix(f"{folder_name}/{line['pose']}", line['start'], line['end'])
            pose = pose.filled(fill_value=0)
            utt_id = line['pose'].split('.')[0]
            utt_id = f"{utt_id}({line['start']})"
            dataset.append((utt_id, pose, fsw_cut(swu2fsw(line['text']))))

    return dataset


def extract_to_matrix(pose_data, output_path, overwrite: bool = False):
    if output_path is not None and output_path.is_file() and not overwrite:
        return np.load(output_path.as_posix())
    if output_path is not None:
        np.save(output_path.as_posix(), pose_data)
        assert output_path.is_file(), output_path
    return pose_data
