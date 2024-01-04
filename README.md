# SignWriting Transcription

This project aims to automatically transcribe SignWriting from isolated/continuous sign language videos. signwriting-transcription adapts the JoeyNMT framework to develop a Neural Machine Translation (NMT) model for converting human poses videos into SignWriting, a visual writing system for sign languages. The project focuses on customizing the vocabulary file, evaluation method, training, and architecture adaptation to achieve accurate translation of diverse human poses into SignWriting symbols. The primary goal is to create a robust NMT model capable of effectively learning and generating SignWriting sequences during the training process.


## Key Modifications

* Preprocessing:
The preprocessing step involves altering the data by normalizing the video poses and transforming them into enumerated vectors. The dataset is then divided into three sets: training, validation, and test, each with its corresponding TSV file.

* Vocabulary:
We have tailored the vocabulary file to accommodate the distinct symbols and gestures associated with SignWriting. This adjustment ensures that the NMT model can effectively learn and generate SignWriting representations during the training process.

* Evaluation Method Enhancement:
The evaluation method for assessing the performance of the NMT model has been customized to align with the intricacies of translating human poses to SignWriting. This involves creating specialized metrics and criteria to measure the accuracy and fluency of the generated SignWriting sequences. More information is specified in the signwriting-evaluation repository.



### Examples

(These examples are taken from the DSGS Vokabeltrainer)

|              |                                   00004                                    |                                   00007                                    |                                   00015                                    |
|:------------:|:--------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
|    Video     |  <img src="assets/examples/00004.gif" width="150px">   |  <img src="assets/examples/00007.gif" width="150px">   |  <img src="assets/examples/00015.gif" width="150px">   |
| SignWriting  |   <img src="assets/examples/00004.png" width="50px">   |   <img src="assets/examples/00007.png" width="50px">   |   <img src="assets/examples/00015.png" width="50px">   |


## Tokenization

SignWriting can be tokenized using the [SignWriting Tokenizer](https://github.com/sign-language-processing/signbank-plus/blob/main/signbank_plus/signwriting/signwriting_tokenizer.py).

## Data

For this study, there are two notable lexicons, containing isolated sign language videos with SignWriting transcriptions.

- [Sign2MINT](https://sign2mint.de/) is a lexicon of German Signed Language (DGS) focusing on natural science subjects. It features 5,263 videos with SignWriting transcriptions. 
- [SignSuisse](https://signsuisse.sgb-fss.ch/) is a Swiss Signed Languages Lexicon that covers Swiss-German Sign Language (DSGS), French Sign Language (LSF), and Italian Sign Language (LIS). The lexicon includes approximately 4,500 LSF videos with [SignWriting transcriptions in SignBank](https://www.signbank.org/signpuddle2.0/index.php?ui=4&sgn=49).

(can also add around 2300 videos from the Vokabeltrainer)


## Installation

Clone the repository:
```
git clone https://github.com/sign-language-processing/signwriting-transcription.git
```
change your working directory to signwriting-transcription
```
cd signwriting-transcription
```
install dependencies using pyproject.toml (make sure that you have pip installed)
```
pip install .
```


## Data Preparation

Download the pose-sign language dataset from the using this commend:
```
!wget -O transcription.zip "https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Ftranscription.zip?alt=media"
```
Unzip the downloaded dataset:
```
unzip transcription.zip -d DataSet
```
Preprocess the data using the provided script:
```
python data_preparation/data_preprocessing.py --srcDir /content/DataSet --trgDir /content/NormData
```
Prepare the data for JoeyNMT using the provided script:
```
python data_preparation/prepare_poses.py --data_root /content/output --dataset_root /content/DataSet --dataset_name poses --tokenizer_type pose-bpe
```

## Training

Create your own configuration file for training or use the config.py for generate it with defult setting

```
python data_preparation/config.py --data-path [input your data path] --experiment-dir [input your experiment dir]
```
Start the training process:

```
python training/train.py /content/output/poses/config.yaml
```


## Tensorboard Visualization

Launch TensorBoard to visualize training progress:
```
tensorboard --logdir /content/models/poses/tensorboard
```


## License

This project is licensed under the MIT License.
