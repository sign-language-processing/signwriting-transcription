# Pose to SignWriting

We adapt the JoeyNMT framework to develop a Neural Machine Translation (NMT) model for
transcribing human poses to SignWriting, a visual writing system for sign languages.

## Usage
```bash
git clone https://github.com/sign-language-processing/signwriting-transcription.git
cd signwriting-transcription
pip install .[pose_to_signwriting]
```
To transcribe a .pose file using the SignWriting FSW format:
```bash
pose_to_signwriting --pose="example.pose" --elan="example.eaf" [--model="{model_number}.ckpt"]
```

## Key Modifications

* Preprocessing:
  The preprocessing step involves altering the data by normalizing the poses and transforming them into enumerated
  vectors. The dataset is then divided into three sets: training, validation, and test, each with its corresponding TSV
  file.

* Vocabulary:
  We have tailored the vocabulary file to accommodate the distinct symbols and gestures associated with SignWriting.
  This adjustment ensures that the NMT model can effectively learn and generate SignWriting representations during the
  training process.

* Evaluation Method Enhancement:
  The evaluation method for assessing the performance of the NMT model has been customized to align with the intricacies
  of translating human poses to SignWriting. This involves creating specialized metrics and criteria to measure the
  accuracy and fluency of the generated SignWriting sequences. More information is specified in the
  signwriting-evaluation repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/sign-language-processing/signwriting-transcription.git
```

change your working directory to signwriting-transcription

```bash
cd signwriting-transcription
```

install dependencies using pyproject.toml (make sure that you have pip installed)

```bash
pip install .
```

## Data Preparation

Download the pose-sign language dataset from the using this commend:

```bash
wget -O transcription.zip "https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Ftranscription.zip?alt=media"
```

Unzip the downloaded dataset:

```bash
unzip transcription.zip -d transcription_data_set
```

Preprocess the data using the provided script:

```bash
python signwriting_transcription/pose_to_signwriting/data/preprocessing.py --src-dir transcription_data_set --trg-dir normalized_data_set
```

Prepare the data for JoeyNMT using the provided script:

```bash
cp data/data.csv normalized_data_set/target.csv
python signwriting_transcription/pose_to_signwriting/data/prepare_poses.py \
  --dataset-root normalized_data_set \
  --data-root vectorized_data_set \
  --dataset-name poses \
  --tokenizer-type pose-bpe
```

## Training

Create your own configuration file for training or use the config.py for generate it with default setting

```bash
python signwriting_transcription/pose_to_signwriting/data/config.py --data-path vectorized_data_set/poses --experiment-dir experiment
```

Start the training process:

```bash
python signwriting_transcription/pose_to_signwriting/joeynmt_pose/training.py experiment/config.yaml
```

## Tensorboard Visualization

Launch TensorBoard to visualize training progress:

```bash
tensorboard --logdir /content/models/poses/tensorboard
```

## using bash script for fast training and updating the model

For training pre-trained model, you can use the following bash script:

```bash
bash signwriting_transcription/pose_to_signwriting/pretrained_model.sh
```

For activating the fine-tuning process, you can use the following bash script:

```bash
bash signwriting_transcription/pose_to_signwriting/fine_tuning.sh pretrain
```

For uploading the model to the cloud, you can use the following bash script:

```bash
bash signwriting_transcription/pose_to_signwriting/upload_model.sh {'pretrain if' it is pre-trained model or nothing if it is fine-tuned model}
```
```