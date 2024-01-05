# coding: utf-8
"""
Training module
"""

from data import load_pose_data

import argparse
import logging
import math
import shutil
import time
from pathlib import Path

from torch.utils.data import Dataset

from joeynmt.helpers import (
    check_version,
    load_config,
    log_cfg,
    make_logger,
    make_model_dir,
    set_seed,
    store_attention_plots,
    write_list_to_file,
)
from joeynmt.model import Model, build_model
from prediction import predict, test
from joeynmt.training import TrainManager
logger = logging.getLogger(__name__)


class PoseTrainManager(TrainManager):
    def __init__(self, model: Model, cfg: dict) -> None:
        super().__init__(model, cfg)
        if not cfg['training'].get('early_stopping_metric', None):
            self.early_stopping_metric = 'fsw_eval'

    def _validate(self, valid_data: Dataset):
        if valid_data.random_subset > 0:  # subsample validation set each valid step
            try:
                valid_data.reset_random_subset()
                valid_data.sample_random_subset(seed=self.stats.steps)
                logger.info(
                    "Sample random subset from dev set: n=%d, seed=%d",
                    len(valid_data),
                    self.stats.steps,
                )
            except AssertionError as e:
                logger.warning(e)

        valid_start_time = time.time()
        (
            valid_scores,
            valid_references,
            valid_hypotheses,
            valid_hypotheses_raw,
            valid_sequence_scores,  # pylint: disable=unused-variable
            valid_attention_scores,
        ) = predict(
            model=self.model,
            data=valid_data,
            compute_loss=True,
            device=self.device,
            n_gpu=self.n_gpu,
            normalization=self.normalization,
            cfg=self.valid_cfg,
            fp16=self.fp16,
        )
        valid_duration = time.time() - valid_start_time

        # for eval_metric in ['loss', 'ppl', 'acc'] + self.eval_metrics:
        for eval_metric, score in valid_scores.items():
            if not math.isnan(score):
                self.tb_writer.add_scalar(f"valid/{eval_metric}", score,
                                          self.stats.steps)

        ckpt_score = valid_scores[self.early_stopping_metric]

        if self.scheduler_step_at == "validation":
            self.scheduler.step(metrics=ckpt_score)

        # update new best
        new_best = self.stats.is_best(ckpt_score)
        if new_best:
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info(
                "Hooray! New best validation result [%s]!",
                self.early_stopping_metric,
            )

        # save checkpoints
        is_better = (self.stats.is_better(ckpt_score, self.ckpt_queue)
                     if len(self.ckpt_queue) > 0 else True)
        if self.num_ckpts < 0 or is_better:
            self._save_checkpoint(new_best, ckpt_score)

        # append to validation report
        self._add_report(valid_scores=valid_scores, new_best=new_best)

        self._log_examples(
            references=valid_references,
            hypotheses=valid_hypotheses,
            hypotheses_raw=valid_hypotheses_raw,
            data=valid_data,
        )

        # store validation set outputs
        write_list_to_file(self.model_dir / f"{self.stats.steps}.hyps",
                           valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(
                attentions=valid_attention_scores,
                targets=valid_hypotheses_raw,
                sources=valid_data.get_list(lang=valid_data.src_lang, tokenized=True),
                indices=self.log_valid_sents,
                output_prefix=(self.model_dir / f"att.{self.stats.steps}").as_posix(),
                tb_writer=self.tb_writer,
                steps=self.stats.steps,
            )

        return valid_duration


def train(cfg_file: str, skip_test: bool = False) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    :param skip_test: whether a test should be run or not after training
    """
    # read config file
    cfg = load_config(Path(cfg_file))

    # make logger
    model_dir = make_model_dir(
        Path(cfg["training"]["model_dir"]),
        overwrite=cfg["training"].get("overwrite", False),
    )
    pkg_version = make_logger(model_dir, mode="train")
    # TODO: save version number in model checkpoints
    if "joeynmt_version" in cfg:
        check_version(pkg_version, cfg["joeynmt_version"])

    # write all entries of config to the log
    log_cfg(cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    src_vocab, trg_vocab, train_data, dev_data, test_data = load_pose_data(
        data_cfg=cfg["data"])

    # store the vocabs and tokenizers
    if src_vocab is not None:
        src_vocab.to_file(model_dir / "src_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.src_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.src_lang].copy_cfg_file(model_dir)
    trg_vocab.to_file(model_dir / "trg_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.trg_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.trg_lang].copy_cfg_file(model_dir)

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = PoseTrainManager(model=model, cfg=cfg)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if not skip_test:
        # predict with the best model on validation and test
        # (if test data is available)

        ckpt = model_dir / f"{trainer.stats.best_ckpt_iter}.ckpt"
        output_path = model_dir / f"{trainer.stats.best_ckpt_iter:08d}.hyps"

        datasets_to_test = {
            "dev": dev_data,
            "test": test_data,
            "src_vocab": src_vocab,
            "trg_vocab": trg_vocab,
        }
        test(
            cfg_file,
            ckpt=ckpt.as_posix(),
            output_path=output_path.as_posix(),
            datasets=datasets_to_test,
        )
    else:
        logger.info("Skipping test after training.")


def main():
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    train(cfg_file=args.config)


if __name__ == "__main__":
    main()
