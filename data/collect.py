from csv import DictReader
from pathlib import Path
import zipfile

from tqdm import tqdm

data_csv = "data.csv"
data = list(DictReader(open(data_csv, "r")))

poses_location = Path("/scratch/amoryo/poses/sign-mt-poses")

unique_poses = set(datum["pose"] for datum in data)
# Create a zip file with all the poses
with zipfile.ZipFile("poses.zip", "w") as poses_zip:
    for pose_name in tqdm(unique_poses):
        poses_zip.write(poses_location / pose_name, pose_name)
