import argparse
from pathlib import Path

from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, correct_wrists, reduce_holistic
from tqdm import tqdm


def preprocess(src_dir, trg_dir):
    src_dir = Path(src_dir)
    trg_dir = Path(trg_dir)
    trg_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(src_dir.glob("*.pose")):
        with open(src_dir / path.name, 'rb') as pose_file:
            pose = Pose.read(pose_file.read())
        pose = reduce_holistic(pose)
        correct_wrists(pose)
        pose = pose.normalize(pose_normalization_info(pose.header))
        with open(trg_dir / path.name, 'wb') as pose_file:
            pose.write(pose_file)


def main():
    print("Preprocessing the data ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, type=str)
    parser.add_argument("--trg-dir", required=True, type=str)
    args = parser.parse_args()
    preprocess(args.src_dir, args.trg_dir)
    print("Done ...")

if __name__ == "__main__":
    main()
