# coding: utf-8
"""
Training module
"""
import argparse
import logging
import shutil

from pathlib import Path

from pose_data import load_pose_data
from joeynmt.helpers import (
    check_version,
    load_config,
    log_cfg,
    make_logger,
    make_model_dir,
    set_seed,
)
from joeynmt.model import build_model
from joeynmt.prediction import test
from joeynmt.training import TrainManager

logger = logging.getLogger(__name__)


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
    trainer = TrainManager(model=model, cfg=cfg)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    train(cfg_file=args.config)