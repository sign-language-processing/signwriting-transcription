# SignWriting Transcription

This project aims to automatically transcribe SignWriting from isolated/continuous sign language videos.

### Data

The full data is available in [sign/data](https://github.com/sign/data/tree/main/signwriting-transcription). These examples are taken from the DSGS Vokabeltrainer:

|             |                                                                    00004                                                                     |                                                                    00007                                                                     |                                                                    00015                                                                     |
|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
|    Video    | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00004.gif?raw=true" width="150px"> | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00007.gif?raw=true" width="150px"> | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00015.gif?raw=true" width="150px"> |
| SignWriting | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00004.png?raw=true" width="50px">  | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00007.png?raw=true" width="50px">  | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00015.png?raw=true" width="50px">  |


## Usage

#### Installation
```shell
conda create --name=multimodalhugs python=3.11 -y 
conda activate multimodalhugs
cd ~/sign-language/signwriting-transcription
pip install .
```

#### Data Preparation

```shell
sbatch ./signwriting_transcription/prepare_data.sh
# Writing /scratch/amoryo/tmp/signwriting-transcription/data.train.tsv
# Writing /scratch/amoryo/tmp/signwriting-transcription/data.dev.tsv
# Writing /scratch/amoryo/tmp/signwriting-transcription/data.test.tsv
# Writing /scratch/amoryo/tmp/signwriting-transcription/data.tokens.txt
```

#### MultimodalHugs Train

```shell
sbatch ./signwriting_transcription/train.sh
```

## Implementations

- [main](https://github.com/sign-language-processing/signwriting-transcription/tree/main) - Latest implementation.
- [v1.0.0](https://github.com/sign-language-processing/signwriting-transcription/tree/1.0.0) - Custom JoeyNMT implementation.