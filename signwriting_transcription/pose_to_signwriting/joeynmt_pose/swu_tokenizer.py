# coding: utf-8
"""
Tokenizer module
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from joeynmt.helpers import ConfigurationError
from joeynmt.tokenizers import (
    BasicTokenizer,
    SentencePieceTokenizer,
    SubwordNMTTokenizer,
    FastBPETokenizer,
    SpeechProcessor
)
from signwriting.tokenizer.signwriting_tokenizer import SignWritingTokenizer

logger = logging.getLogger(__name__)


class SwuTokenizer(BasicTokenizer):
    """
    Tokenizer for the SWU (SignWriting Unicode) language.

    This tokenizer is derived from BasicTokenizer and customized for SWU.
    It provides functionality to tokenize SWU text.
    """

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
        tokenizer = SpeechProcessor(
            level=cfg["level"],
            num_freq=cfg["num_freq"],
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
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
