import numpy as np
from pose_format import Pose
import pandas as pd
from swu_representation import swu2data

FrameRate = 29.97003


def ms2frame(ms) -> int:
    return int(ms / 1000 * FrameRate)


def pose_to_matrix(file_path, start_ms, end_ms):
    with open(file_path, "rb") as f:
        pose = Pose.read(f.read())
    pose = pose.body.data
    pose = pose.reshape(pose.shape[0], pose.shape[2] * pose.shape[3])
    pose = pose[ms2frame(start_ms):ms2frame(end_ms)]
    return pose


def load_dataset(folder_name):

    target = pd.read_csv(f'{folder_name}/target.csv')
    dataset = []
    for line in target.values:
        pose = pose_to_matrix(f'{folder_name}/{line[0]}', line[2], line[3])
        pose = pose.filled(fill_value=0)
        utt_id = line[0].split('.')[0]
        utt_id = f'{utt_id}({line[2]})'
        dataset.append((utt_id, pose, swu2data(line[4])))
    return dataset


def extract_to_fbank(pose_data, output_path, overwrite: bool = False):
    if output_path is not None and output_path.is_file() and not overwrite:
        return np.load(output_path.as_posix())
    if output_path is not None:
        np.save(output_path.as_posix(), pose_data)
        assert output_path.is_file(), output_path
    return pose_data


if __name__ == "__main__":
    dataSet = load_dataset("Dataset")

    print(dataSet)
