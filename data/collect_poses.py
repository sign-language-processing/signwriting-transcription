from csv import DictReader
from pathlib import Path
import zipfile

from tqdm import tqdm

with open("data.csv", "r", encoding="utf-8") as f:
    data = list(DictReader(f))

poses_location = Path("/Volumes/Echo/GCS/sign-mt-poses")

unique_poses = set(datum["pose"] for datum in data)
# Create a zip file with all the poses
with zipfile.ZipFile("/Volumes/Echo/GCS/sign-language-datasets/poses/holistic/transcription.zip", "w") as poses_zip:
    for pose_name in tqdm(unique_poses):
        poses_zip.write(poses_location / pose_name, pose_name)
