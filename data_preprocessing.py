import argparse
from pathlib import Path

from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, correct_wrists, reduce_holistic


def preprocess(srcDir, trgDir):
    srcDir = Path(srcDir)
    trgDir = Path(trgDir)
    trgDir.mkdir(parents=True, exist_ok=True)
    for path in srcDir.iterdir():
        if path.is_file() and path.suffix == ".pose":
            with open(srcDir / path.name, 'rb') as pose_file:
                pose = Pose.read(pose_file.read())
            pose = reduce_holistic(pose)
            correct_wrists(pose)
            pose = pose.normalize(pose_normalization_info(pose.header))
            with open(trgDir / path.name, 'w+b') as pose_file:
                pose.write(pose_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcDir", required=True, type=str)
    parser.add_argument("--trgDir", required=True, type=str)
    args = parser.parse_args()
    preprocess(args.srcDir, args.trgDir)


if __name__ == "__main__":
    main()
