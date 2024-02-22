import argparse
from pathlib import Path

from pose_format import Pose
from pose_format.utils.generic import reduce_holistic
from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
from tqdm import tqdm


def preprocess_single_file(src_file: Path, normalization=True):
    with open(src_file, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
    if normalization:
        pose = pre_process_mediapipe(pose)
        pose = normalize_mean_std(pose)
    else:
        pose = reduce_holistic(pose)

    return pose


def preprocess(src_dir: Path, trg_dir: Path, normalization=True):
    src_dir = Path(src_dir)
    trg_dir = Path(trg_dir)
    trg_dir.mkdir(parents=True, exist_ok=True)
    for src_file in tqdm(src_dir.glob("*.pose")):
        trg_file = trg_dir / src_file.name
        preprocessed_pose = preprocess_single_file(src_file, normalization)
        with open(trg_file, 'wb') as pose_file:
            preprocessed_pose.write(pose_file)


def main():
    print("Preprocessing the data ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, type=str)
    parser.add_argument("--trg-dir", required=True, type=str)
    parser.add_argument("--normalization", required=False, type=str, default="True")
    args = parser.parse_args()
    args.normalization = args.normalization == "True"
    preprocess(args.src_dir, args.trg_dir, args.normalization)
    print("Done ...")


if __name__ == "__main__":
    main()
