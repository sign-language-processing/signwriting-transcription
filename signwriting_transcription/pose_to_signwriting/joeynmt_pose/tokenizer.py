# coding: utf-8
"""
Tokenizer module
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from joeynmt.helpers import ConfigurationError

from joeynmt.tokenizers import (
    BasicTokenizer,
    SentencePieceTokenizer,
    SubwordNMTTokenizer,
    FastBPETokenizer,
    SpeechProcessor
)
from pose_format.numpy import NumPyPoseBody
from signwriting.tokenizer.signwriting_tokenizer import SignWritingTokenizer
from signwriting_transcription.pose_to_signwriting.data.datasets_pose import pose_ndarray_to_matrix

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,too-many-instance-attributes
class PoseProcessor(SpeechProcessor):
    def __init__(
            self,
            level: str = "pose",
            num_freq: int = 534,
            normalize: bool = False,
            max_length: int = -1,
            min_length: int = -1,
            augment: bool = False,
            aug_param: float = 0.2,
            noise: bool = False,
            noise_param: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.level = level
        self.num_freq = num_freq
        self.normalize = normalize
        self.augment = augment
        self.aug_param = aug_param
        self.noise = noise
        self.noise_param = noise_param

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

        self.root_path = ""  # assigned later by dataset.__init__()

    def get_features(self, pose_path: str, buffer_size: float) -> np.ndarray:
        pose_path, *extra = pose_path.split(":")
        _, *extra = pose_path.split('.')[0].split(',')
        root_path = Path(self.root_path)
        _path = root_path / pose_path
        if not _path.is_file():
            raise FileNotFoundError(f"File not found: {_path}")
        if _path.suffix != ".npy":
            raise ValueError(f"Invalid file type: {_path}")

        if len(extra) == 0:
            features = np.load(_path.as_posix())
            features = pose_ndarray_to_matrix(features, 0, 29.97003)    # 29.97003 default fps
        elif len(extra) == 5:
            start, end, fps, last_segment, next_segment = extra
            start_addition = (start - last_segment) * buffer_size
            end_reduction = (next_segment - end) * buffer_size
            features = np.load(_path.as_posix())
            features = pose_ndarray_to_matrix(features, start - start_addition, fps, end + end_reduction)
        else:
            raise ValueError("Invalid format")

        assert len(features.shape) == 2, "spectrogram must be a 2-D array."
        return features

    def __call__(self, line: str, is_train: bool = False) -> np.ndarray:
        """
        get features

        :param line: path to audio file or pre-extracted features
        :param is_train:

        :return: spectrogram in shape (num_frames, num_freq)
        """
        # lined is normalised by media-pipe normalisation
        buffer = random.random() * 0.5
        item = self.get_features(line, buffer)  # shape = (num_frames, num_freq)

        num_frames, num_freq = item.shape
        assert num_freq == self.num_freq

        if self._filter_too_short_item(num_frames):
            # A too short sequence cannot be convolved!
            # -> filter out anyway even in test-dev set.
            return None
        if self._filter_too_long_item(num_frames):
            # Don't use too long sequence in training.
            if is_train:  # pylint: disable=no-else-return
                return None
            else:  # in test, truncate the sequence
                item = item[:self.max_length, :]
                num_frames = item.shape[0]
                assert num_frames <= self.max_length

        # cmvn / specaugment
        # pylint: disable=not-callable

        if self.augment:
            item = item.reshape(item.shape[0], 1, -1, 3)
            body = NumPyPoseBody(None, item, np.ones(item.shape[:3]))
            rot_std, she_std, sca_std = np.random.uniform(-1 * self.aug_param, self.aug_param, 3)
            item = body.augment2d(rotation_std=rot_std, shear_std=she_std, scale_std=sca_std)
            item = item.data.reshape(item.data.shape[0], -1)
            item.filled(fill_value=0)
        if self.noise:
            gaussian_noise = np.random.normal(0, self.noise_param, item.shape)
            item = item + gaussian_noise
        return item

    def _filter_too_short_item(self, length: int) -> bool:
        return self.min_length > length > 0

    def _filter_too_long_item(self, length: int) -> bool:
        return length > self.max_length > 0

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"level={self.level}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), ")


# pylint: disable=too-many-branches
class SwuTokenizer(BasicTokenizer):

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            level: str = "vpf",
            lowercase: bool = False,
            normalize: bool = False,
            max_length: int = -1,
            min_length: int = -1,
            **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "vpf"
        kwargs = {'init_token': BOS_TOKEN,
                  'eos_token': EOS_TOKEN,
                  'pad_token': PAD_TOKEN,
                  'unk_token': UNK_TOKEN}
        self.tokenizer = SignWritingTokenizer(starting_index=None, **kwargs)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        tokenized = [self.tokenizer.i2s[s] for s in self.tokenizer.tokenize(raw_input)]
        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self,
                     sequence: Union[List[str], str],
                     generate_unk: bool = False) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            sequence = self._remove_special(sequence, generate_unk=generate_unk)
            # Decode back to str
            sequence = [self.tokenizer.s2i[s] for s in sequence]
            sequence = self.tokenizer.detokenize(sequence)
        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def copy_cfg_file(self, model_dir: Path) -> None:
        pass

    def sign2id(self, sign: str) -> int:
        return self.tokenizer.s2i[sign]

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"pretokenizer={self.pretokenizer}, "
                f"tokenizer=SignWritingTokenizer)")


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    """Builds tokenizer."""
    tokenizer = None
    tokenizer_cfg = cfg.get("tokenizer_cfg", {})

    # assign lang for moses tokenizer
    if tokenizer_cfg.get("pretokenizer", "none") == "moses":
        tokenizer_cfg["lang"] = cfg["lang"]

    if cfg["level"] in ["word", "char"]:
        tokenizer = BasicTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
    elif cfg["level"] == "vpf":
        tokenizer = SwuTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", cfg.get("bpe_type", "sentencepiece"))
        if tokenizer_type == "sentencepiece":
            assert "model_file" in tokenizer_cfg
            tokenizer = SentencePieceTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "subword-nmt":
            assert "codes" in tokenizer_cfg
            tokenizer = SubwordNMTTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "fastbpe":
            assert "codes" in tokenizer_cfg
            tokenizer = FastBPETokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        else:
            raise ConfigurationError(f"{tokenizer_type}: Unknown tokenizer type.")
    elif cfg["level"] == "frame":
        tokenizer_type = cfg.get("tokenizer_type", cfg.get("bpe_type", "pose"))
        if tokenizer_type == "speech":
            tokenizer = SpeechProcessor(
                level=cfg["level"],
                num_freq=cfg["num_freq"],
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "pose":
            tokenizer = PoseProcessor(
                level=cfg["level"],
                num_freq=cfg["num_freq"],
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                augment=cfg.get("augment", True),
                aug_param=cfg.get("aug_param", 0.2),
                noise=cfg.get("noise", False),
                noise_param=cfg.get("noise_param", 0.1),
                **tokenizer_cfg,
            )
        else:
            raise ConfigurationError(f"{tokenizer_type}: Unknown tokenizer type.")
    else:
        raise ConfigurationError(f"{cfg['level']}: Unknown tokenization level.")
    return tokenizer


def build_tokenizer(data_cfg: Dict) -> Dict[str, BasicTokenizer]:
    task = data_cfg.get("task", "MT").upper()
    src_lang = data_cfg["src"]["lang"] if task == "MT" else "src"
    trg_lang = data_cfg["trg"]["lang"] if task == "MT" else "trg"
    tokenizer = {
        src_lang: _build_tokenizer(data_cfg["src"]),
        trg_lang: _build_tokenizer(data_cfg["trg"]),
    }
    logger.info("%s Tokenizer: %s", src_lang, tokenizer[src_lang])
    logger.info("%s Tokenizer: %s", trg_lang, tokenizer[trg_lang])
    return tokenizer
