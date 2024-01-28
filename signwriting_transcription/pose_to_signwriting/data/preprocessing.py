import argparse
from pathlib import Path

from pose_format.utils.generic import reduce_holistic
from pose_format import Pose
from tqdm import tqdm
from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std


def preprocess(src_dir, trg_dir, action=True):
    src_dir = Path(src_dir)
    trg_dir = Path(trg_dir)
    trg_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(src_dir.glob("*.pose")):
        with open(src_dir / path.name, 'rb') as pose_file:
            pose = Pose.read(pose_file.read())
            pose = reduce_holistic(pose)
        if action:
            pose = pre_process_mediapipe(pose)
            pose = normalize_mean_std(pose)
        with open(trg_dir / path.name, 'wb') as pose_file:
            pose.write(pose_file)


def main():
    print("Preprocessing the data ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, type=str)
    parser.add_argument("--trg-dir", required=True, type=str)
    parser.add_argument("--action", required=False, type=str, default="True")
    args = parser.parse_args()
    args.action = args.action == "True"
    preprocess(args.src_dir, args.trg_dir, args.action)
    print("Done ...")


if __name__ == "__main__":
    main()
