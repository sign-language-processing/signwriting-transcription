# Create the config
import argparse
from pathlib import Path


def create_config(data_path="/output/poses", experiment_dir='/model/poses'):
    data_path = Path(data_path)
    experiment_dir = Path(experiment_dir)

    config = """
    name: "poses"
    joeynmt_version: 2.0.0

    data:
        task: "S2T"                     # "S2T" for speech-to-text, "MT" for (text) translation
        train: "{data_dir}/train"
        dev:   "{data_dir}/dev"
        test:  "{data_dir}/test"
        dataset_type: "speech"          # SpeechDataset takes tsv as input
        src:
            lang: "en_ng"
            num_freq: 534                # number of frequencies of audio inputs
            max_length: 3000            # much longer than text sequence!
            min_length: 10              # have to be specified so that 1d-conv works!
            level: "frame"              # Here we specify we're working on BPEs.
            tokenizer_type: "pose"
            augment: True
            aug_param: 0.2
            noise: False
            noise_param: 0.1
            tokenizer_cfg: 
                specaugment:
                    freq_mask_n: 1
                    freq_mask_f: 5
                    time_mask_n: 1
                    time_mask_t: 10
                    time_mask_p: 1.0
                cmvn:
                    norm_means: True
                    norm_vars: True
                    before: True
        trg:
            lang: "en_ng"
            max_length: 100
            lowercase: False
            level: "vpf"                # Here we specify we're working on BPEs.
            voc_file: "{data_dir}/spm_bpe1182.vocab"
            tokenizer_type: "pose-vpf"
            tokenizer_cfg: 
                model_file: "{data_dir}/spm_bpe1182.model"
                pretokenize: "none"

    testing:
        eval_all_metrics: False
        n_best: 1
        beam_size: 5
        beam_alpha: 1.0
        batch_size: 4
        batch_type: "sentence"
        max_output_length: 100          # Don't generate translations longer than this.
        # eval_metrics: ["wer"]           # Use "wer" for ASR task, "bleu" for ST task
        sacrebleu_cfg:                  # sacrebleu options
            tokenize: "intl"            # `tokenize` option in sacrebleu.corpus_bleu() function (options include: "none" (use for already tokenized test data), "13a" (default minimal tokenizer), "intl" which mostly does punctuation and unicode, etc) 

    training:
        #load_model: "{experiment_dir}/1.ckpt" # if uncommented, load a pre-trained model from this checkpoint
        random_seed: 42
        optimizer: "adam"
        normalization: "tokens"
        adam_betas: [0.9, 0.98] 
        scheduling: "plateau"
        patience: 15
        learning_rate: 0.0002
        learning_rate_min: 0.00000001
        weight_decay: 0.0
        label_smoothing: 0.1
        loss: "crossentropy-ctc"       # use CrossEntropyLoss + CTCLoss
        ctc_weight: 0.3                # ctc weight in interpolation
        batch_size: 4                  # much bigger than text! your "tokens" are "frames" now.
        batch_type: "sentence"
        batch_multiplier: 1
        # early_stopping_metric:       # by default, early stopping uses "fsw_eval" metric
        epochs: 15                     # Decrease for when playing around and checking of working.
        validation_freq: 1000          # Set to at least once per epoch.
        logging_freq: 100
        model_dir: "{experiment_dir}"
        overwrite: True
        shuffle: True
        use_cuda: True
        print_valid_sents: [0, 1, 2, 3]
        keep_best_ckpts: 2

    model:
        initializer: "xavier_uniform"
        bias_initializer: "zeros"
        init_gain: 1.0
        embed_initializer: "xavier_uniform"
        embed_init_gain: 1.0
        tied_embeddings: False       # DIsable embeddings sharing between enc(audio) and dec(text)
        tied_softmax: False
        encoder:
            type: "transformer"
            num_layers: 12           # Common to use doubly bigger encoder than decoder in S2T.
            num_heads: 4
            embeddings:
                embedding_dim: 534    # Must be same as the frequency of the filterbank features!
            # typically ff_size = 4 x hidden_size
            hidden_size: 256
            ff_size: 1024
            dropout: 0.1
            layer_norm: "pre"
            # new for S2T:
            subsample: True           # enable 1d conv module
            conv_kernel_sizes: [5, 5] # convolution kernel sizes (window width)
            conv_channels: 512        # convolution channels
            in_channels: 534           # Must be same as the embedding_dim
        decoder:
            type: "transformer"
            num_layers: 6
            num_heads: 4
            embeddings:
                embedding_dim: 256
                scale: True
                dropout: 0.0
            # typically ff_size = 4 x hidden_size
            hidden_size: 256
            ff_size: 1024
            dropout: 0.1
            layer_norm: "pre"
    """.format(data_dir=data_path.as_posix(),
               experiment_dir=experiment_dir.as_posix())

    (data_path / 'config.yaml').write_text(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", required=True, type=str)
    parser.add_argument("--experiment-dir", "-e", required=True, type=str)
    args = parser.parse_args()
    create_config(args.data_path, args.experiment_dir)


if __name__ == '__main__':
    main()
