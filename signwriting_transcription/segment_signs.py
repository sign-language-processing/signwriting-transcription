import argparse
from collections import Counter
from csv import DictReader, DictWriter
from pathlib import Path

from pose_format import Pose
from sign_language_segmentation.bin import segment_pose
from tqdm import tqdm


def get_segmented_datum(datum, poses_dir: Path):
    file_path = poses_dir / datum["pose"]
    with open(file_path, "rb") as f:
        pose = Pose.read(f)

    eaf, _ = segment_pose(pose, verbose=False)
    sign_annotations = eaf.get_annotation_data_for_tier('SIGN')

    if len(sign_annotations) == 0:
        raise ValueError(f"No sign annotations found for {datum['pose']}")

    new_datum = datum.copy()
    new_datum["start"] = sign_annotations[0][0]
    new_datum["end"] = sign_annotations[-1][1]
    return new_datum


def augmented_data(data, poses_dir: Path):
    for datum in tqdm(data):
        # We try to create a new sign segment entry if the current entry covers the entire video
        if int(datum["start"]) == 0 and int(datum["end"]) < 10000:
            try:
                yield get_segmented_datum(datum, poses_dir)
            except Exception as e:
                print("Skipping", datum["pose"], e)

        yield datum


def datum_index(datum):
    return "_".join(f"{k}:{v}" for k, v in datum.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = list(DictReader(f))

    # if output exists, remove it from source and append
    is_append = args.output.exists()
    if is_append:
        with open(args.output, "r", encoding="utf-8") as f:
            existing_data = {datum_index(d) for d in DictReader(f)}
        data = [d for d in data if datum_index(d) not in existing_data]
        if len(data) == 0:
            return

    with open(args.output, "a" if is_append else "w", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=data[0].keys())
        if not is_append:
            writer.writeheader()

        for i, datum in enumerate(augmented_data(data, args.poses)):
            writer.writerow(datum)
            if i % 100 == 0:
                f.flush()


if __name__ == "__main__":
    main()
