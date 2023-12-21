import numpy as np
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pathlib import Path


def pose_to_matrix(file_path):
    with open(file_path, "rb") as f:
        pose = Pose.read(f.read())
    pose = pose.body.data
    pose = pose.reshape(pose.shape[0], pose.shape[2] * pose.shape[3])
    return pose


def load_dataset(folder_name):
    folder_path = Path(folder_name)
    files = list(folder_path.glob("*.pose"))
    # open target.txt file
    # read each line
    targetFile = open(folder_path / "target.txt", "r")
    dataSet = []
    for file, line in zip(files, targetFile):
        pose = pose_to_matrix(file)
        pose = pose.filled(fill_value=0)
        if line.endswith('\n'):
            line = line[:-1]
        dataSet.append((file.stem, pose, line))
    return dataSet


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
