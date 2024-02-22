# Data

Data includes MediaPipe poses of videos from Sign2MINT and Signsuisse, transcribed using SignWriting.

Data is got from the database using `get_data.py`.


## Poses

Poses are collected using `collect_poses.py` and are available to download from [Google Cloud Storage](https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Ftranscription.zip?alt=media).

It is recommended to pre-process the poses before using them for training. For example:
```python
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, correct_wrists, reduce_holistic

with open('example.pose', 'rb') as pose_file:
    pose = Pose.read(pose_file.read())

# Remove legs, simplify face
pose = reduce_holistic(pose)
# Align hand wrists with body wrists
correct_wrists(pose)
# Adjust pose based on shoulder positions
pose = pose.normalize(pose_normalization_info(pose.header))

# Save normalized pose
with open('example.posebody', 'wb') as pose_file:
    pose.write(pose_file)
```

## Automatic Segmentation

Most annotations come from single sign videos with the annotation spanning the entire video. 
However, in real use cases, we would like to transcribe continuous signing, and training on full single-sign videos might not yield correct results.

We automatically segment the single-sign videos using [sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)
to extract the sign boundary. Where successful, we record the new sign segments in data_segmentation.csv and use them for additional training data.

## Issues

- `.pose` files are not normalized, and are not centered around the origin.
- `.pose` files do not allow `float` fps values, only `int` fps values. 
  Therefore, every annotation that starts at $0$ should be assumed to end at the end.
- `19097be0e2094c4aa6b2fdc208c8231e.pose` comes from [Why SignWriting?](https://www.youtube.com/watch?v=Mtl7dmyHgJU), 
  and demonstrates transcription of continuous sign language. The actual frame rate is `29.970030`.
  Therefore, it should only be used for testing continuous sign language transcription.
