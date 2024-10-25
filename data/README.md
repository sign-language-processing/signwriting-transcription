# Data

Data includes MediaPipe poses of videos from multiple sources, transcribed using SignWriting.

Data is got from the database using `get_data.py`.

### Sources

- ChicagoFSWild - About 50,000 fingerspelled signs. Low quality transcriptions. No specific indicator, except for using only hand symbols.
- dictio - about 36,000 videos. pose files starts with "dictio". Every sign has two videos, one from a direct angle, and one from a side angle (unmarked).
- Sign2MINT - about 5000 isolated signs from Sign2MINT. pose files starts with "s2m".
- SignSuisse - about 4000 isolated signs from SignSuisse. pose files starts with "ss".
- FLEURS-ASL - about 200, extremely high quality continuous sign language transcriptions with detailed facial expressions. pose files starts with "fasl".
- `19097be0e2094c4aa6b2fdc208c8231e.pose` comes from [Why SignWriting?](https://www.youtube.com/watch?v=Mtl7dmyHgJU), and demonstrates transcription of continuous sign language.

## Poses

Poses are collected using `collect_poses.py` and are available to download from [Google Cloud Storage](https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Ftranscription.zip?alt=media).

It is recommended to pre-process the poses when using them for training. For example:
```python
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, correct_wrists, reduce_holistic

# Load full pose video
with open('example.pose', 'rb') as pose_file:
    pose = Pose.read(pose_file.read())

# Or load based on start and end time (faster)
with open('example.pose', 'rb') as pose_file:
    pose = Pose.read(pose_file.read(), start_time=0, end_time=10)
    
# This imo is IDEAL for experimentation, but shouldn't be used for the final model
## Remove legs, simplify face
pose = reduce_holistic(pose)
## Align hand wrists with body wrists
correct_wrists(pose)

# This should be used always
## Adjust pose based on shoulder positions
pose = pose.normalize(pose_normalization_info(pose.header))
```

## Issues to be aware of:

- `.pose` files are not normalized, and are not centered around the origin.

----

Not sure if relevant anymore:

## Automatic Segmentation

Most annotations come from single sign videos with the annotation spanning the entire video. 
However, in real use cases, we would like to transcribe continuous signing, and training on full single-sign videos might not yield correct results.

We automatically segment the single-sign videos using [sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)
to extract the sign boundary. Where successful, we record the new sign segments in data_segmentation.csv and use them for additional training data.
