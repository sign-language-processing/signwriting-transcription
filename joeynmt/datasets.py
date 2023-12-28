# coding: utf-8
"""
Dataset module
"""
import logging
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from joeynmt.batch import Batch
from joeynmt.constants import PAD_ID
from joeynmt.helpers import ConfigurationError, read_list_from_file
from joeynmt.tokenizers import BasicTokenizer, SpeechProcessor

logger = logging.getLogger(__name__)
CPU_DEVICE = torch.device("cpu")


class BaseDataset(Dataset):
    """
    BaseDataset which loads and looks up data.
    - holds pointer to tokenizers, encoding functions.

    :param path: path to data directory
    :param src_lang: source language code, i.e. `en`
    :param trg_lang: target language code, i.e. `de`
    :param has_trg: bool indicator if trg exists
    :param split: bool indicator for train set or not
    :param tokenizer: tokenizer objects
    :param sequence_encoder: encoding functions
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "MT",
    ):
        self.path = path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.has_trg = has_trg
        self.split = split
        if self.split == "train":
            assert self.has_trg

        _place_holder = {self.src_lang: None, self.trg_lang: None}
        self.tokenizer = _place_holder if tokenizer is None else tokenizer
        self.sequence_encoder = (_place_holder
                                 if sequence_encoder is None else sequence_encoder)

        # for ransom subsampling
        self.random_subset = random_subset

        self.task = task

    def sample_random_subset(self, seed: int = 42) -> None:
        # pylint: disable=unused-argument
        assert (
            self.split != "test" and self.__len__() > self.random_subset > 0
        ), f"Can only subsample from train or dev set larger than {self.random_subset}."

    def reset_random_subset(self):
        raise NotImplementedError

    def load_data(self, path: Path, **kwargs) -> Any:
        """
        load data
            - preprocessing (lowercasing etc) is applied here.
        """
        raise NotImplementedError

    def get_item(self, idx: int, lang: str) -> List[str]:
        """
        seek one src/trg item of given index.
            - tokenization is applied here.
            - length-filtering, bpe-dropout etc also triggered if self.split == "train"
        """
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, str]) -> Tuple[List[str], List[str]]:
        """lookup one item pair of given index."""
        src, trg = None, None
        src = self.get_item(idx=idx, lang=self.src_lang)
        if self.has_trg:
            trg = self.get_item(idx=idx, lang=self.trg_lang)
            if trg is None:
                src = None
        return src, trg

    def get_list(self,
                 lang: str,
                 postproccessed: bool = False,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        """get data column-wise."""
        raise NotImplementedError

    @property
    def src(self) -> List[str]:
        """get detokenized preprocessed data in src language."""
        return self.get_list(self.src_lang, postproccessed=False, tokenized=False)

    @property
    def trg(self) -> List[str]:
        """get detokenized preprocessed data in trg language."""
        return self.get_list(self.trg_lang, postproccessed=False, tokenized=False) \
            if self.has_trg else []

    def collate_fn(
        self,
        batch: List[Tuple],
        pad_index: int = PAD_ID,
        device: torch.device = CPU_DEVICE,
    ) -> Batch:
        """
        Custom collate function.
        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn for details.
        Please override the batch class here. (not in TrainManager)

        :param batch:
        :param pad_index:
        :param device:
        :return: joeynmt batch object
        """

        def _is_valid(s, t, has_trg):
            # pylint: disable=no-else-return
            if has_trg:
                return s is not None and t is not None
            else:
                return s is not None

        batch = [(s, t) for s, t in batch if _is_valid(s, t, self.has_trg)]
        src_list, trg_list = zip(*batch)
        assert len(batch) == len(src_list), (len(batch), len(src_list))
        assert all(s is not None for s in src_list), src_list
        src, src_length = self.sequence_encoder[self.src_lang](src_list)

        if self.has_trg:
            assert all(t is not None for t in trg_list), trg_list
            trg, trg_length = self.sequence_encoder[self.trg_lang](trg_list)
        else:
            assert all(t is None for t in trg_list)
            trg, trg_length = None, None

        return Batch(
            src=(torch.tensor(src).long()
                 if self.task == "MT" else torch.tensor(src).float()),
            src_length=torch.tensor(src_length).long(),
            trg=torch.tensor(trg).long() if trg else None,
            trg_length=torch.tensor(trg_length).long() if trg_length else None,
            device=device,
            pad_index=pad_index,
            has_trg=self.has_trg,
            is_train=self.split == "train",
            task=self.task,
        )

    def make_iter(
        self,
        batch_size: int,
        batch_type: str = "sentence",
        seed: int = 42,
        shuffle: bool = False,
        num_workers: int = 0,
        pad_index: int = PAD_ID,
        device: torch.device = CPU_DEVICE,
    ) -> DataLoader:
        """
        Returns a torch DataLoader for a torch Dataset. (no bucketing)

        :param batch_size: size of the batches the iterator prepares
        :param batch_type: measure batch size by sentence count or by token count
        :param seed: random seed for shuffling
        :param shuffle: whether to shuffle the data before each epoch
            (for testing, no effect even if set to True)
        :param num_workers: number of cpus for multiprocessing
        :param pad_index:
        :param device:
        :return: torch DataLoader
        """
        # sampler
        sampler: Sampler[int]  # (type annotation)
        if shuffle and self.split == "train":
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(self, generator=generator)
        else:
            sampler = SequentialSampler(self)

        # batch generator
        if batch_type == "sentence":
            batch_sampler = SentenceBatchSampler(sampler,
                                                 batch_size=batch_size,
                                                 drop_last=False)
        elif batch_type == "token":
            batch_sampler = TokenBatchSampler(sampler,
                                              batch_size=batch_size,
                                              drop_last=False)
        else:
            raise ConfigurationError(f"{batch_type}: Unknown batch type")

        assert self.sequence_encoder[self.src_lang] is not None
        if self.has_trg:
            assert self.sequence_encoder[self.trg_lang] is not None

        # data iterator
        return DataLoader(
            dataset=self,
            batch_sampler=batch_sampler,
            collate_fn=partial(self.collate_fn, pad_index=pad_index, device=device),
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(split={self.split}, len={self.__len__()}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg}, random_subset={self.random_subset})")


class PlaintextDataset(BaseDataset):
    """
    PlaintextDataset which stores plain text pairs.
    - used for text file data in the format of one sentence per line.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "MT",
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
        )

        # load data
        self.data = self.load_data(path, **kwargs)
        self._initial_len = len(self.data[self.src_lang])

        # for random subsampling
        self.idx_map = []

    def load_data(self, path: str, **kwargs) -> Any:

        def _pre_process(seq, lang):
            if self.tokenizer[lang] is not None:
                seq = [self.tokenizer[lang].pre_process(s) for s in seq if len(s) > 0]
            return seq

        path = Path(path)
        src_file = path.with_suffix(f"{path.suffix}.{self.src_lang}")
        assert src_file.is_file(), f"{src_file} not found. Abort."

        src_list = read_list_from_file(src_file)
        data = {self.src_lang: _pre_process(src_list, self.src_lang)}

        if self.has_trg:
            trg_file = path.with_suffix(f"{path.suffix}.{self.trg_lang}")
            assert trg_file.is_file(), f"{trg_file} not found. Abort."

            trg_list = read_list_from_file(trg_file)
            data[self.trg_lang] = _pre_process(trg_list, self.trg_lang)
            assert len(src_list) == len(trg_list)
        return data

    def sample_random_subset(self, seed: int = 42) -> None:
        super().sample_random_subset(seed)  # check validity

        random.seed(seed)  # resample every epoch: seed += epoch_no
        self.idx_map = list(random.sample(range(self._initial_len), self.random_subset))

    def reset_random_subset(self):
        self.idx_map = []

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        line = self._look_up_item(idx, lang)
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line, is_train=is_train)

        if item is None:
            logger.debug("Skip %d-th instance (%s): {%s}", idx, lang, line)
        return item

    def _look_up_item(self, idx: int, lang: str) -> str:
        try:
            if len(self.idx_map) > 0:
                idx = self.idx_map[idx]
            line = self.data[lang][idx]
            return line
        except Exception as e:
            logger.error(idx, self._initial_len)
            raise Exception from e

    def get_list(self,
                 lang: str,
                 postproccessed: bool = False,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Return list of preprocessed sentences in the given language.
        (not length-filtered, no bpe-dropout)
        """
        item_list = []
        for idx in range(self.__len__()):
            item = self._look_up_item(idx, lang)
            if postproccessed:
                item = self.tokenizer[lang].post_process(item)
            elif tokenized:
                item = self.tokenizer[lang](self._look_up_item(idx, lang),
                                            is_train=False)
            item_list.append(item)
        return item_list

    def __len__(self) -> int:
        if len(self.idx_map) > 0:
            return len(self.idx_map)
        return self._initial_len


class TsvDataset(BaseDataset):
    """
    TsvDataset which handles data in tsv format.
    - file_name should be specified without extention `.tsv`
    - needs src_lang and trg_lang (i.e. `en`, `de`) in header.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "MT",
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
        )

        # load tsv file
        self.df = self.load_data(path, **kwargs)

        # for random subsampling
        self._initial_df = None

    def load_data(self, path: str, **kwargs) -> Any:
        path = Path(path)
        file_path = path.with_suffix(f"{path.suffix}.tsv")
        assert file_path.is_file(), f"{file_path} not found. Abort."

        # read tsv data
        try:
            # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
            import pandas as pd

            df = pd.read_csv(
                file_path.as_posix(),
                sep="\t",
                header=0,
                encoding="utf-8",
                escapechar="\\",
                quoting=3,
                na_filter=False,
            )
            # TODO: use `chunksize` for online data loading.
            assert self.src_lang in df.columns
            df[self.src_lang] = df[self.src_lang].apply(
                self.tokenizer[self.src_lang].pre_process)

            if self.trg_lang not in df.columns:
                self.has_trg = False
                assert self.split == "test"
            if self.has_trg:
                df[self.trg_lang] = df[self.trg_lang].apply(
                    self.tokenizer[self.trg_lang].pre_process)
            return df

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def sample_random_subset(self, seed: int = 42) -> None:
        super().sample_random_subset(seed)  # check validity

        if self._initial_df is None:
            self._initial_df = self.df.copy(deep=True)

        self.df = self._initial_df.sample(
            n=self.random_subset,
            replace=False,
            random_state=seed,  # resample every epoch: seed += epoch_no
        ).reset_index()

    def reset_random_subset(self):
        if self._initial_df is not None:
            self.df = self._initial_df
            self._initial_df = None

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        line = self.df.iloc[idx][lang]
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line, is_train=is_train)
        return item

    def get_list(self,
                 lang: str,
                 postproccessed: bool = False,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        if postproccessed:
            sents = self.df[lang].apply(self.tokenizer[lang].post_process).to_list()
        elif tokenized:
            sents = self.df[lang].apply(self.tokenizer[lang]).to_list()
        else:
            sents = self.df[lang].to_list()
        return sents

    def __len__(self) -> int:
        return len(self.df)


class SpeechDataset(TsvDataset):
    """
    Speech Dataset
    """

    def __init__(
        self,
        path: str,
        src_lang: str = "src",
        trg_lang: str = "trg",
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, Union[BasicTokenizer, SpeechProcessor]] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "S2T",
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
        )

        # load tsv file
        self.df = self.load_data(path, **kwargs)

        assert isinstance(self.tokenizer["src"], SpeechProcessor)
        self.tokenizer["src"].root_path = Path(path).parent

        # for random subsampling
        self._initial_df = None

    def load_data(self, path: str, **kwargs) -> Any:
        path = Path(path)
        file_path = path.with_suffix(f"{path.suffix}.tsv")
        assert file_path.is_file(), f"{file_path} not found. Abort."

        # read tsv data
        try:
            # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
            import pandas as pd

            # TODO: use `chunksize` for online data loading.
            dtype = {'id': str, 'src': str, 'trg': str, 'n_frames': int}
            df = pd.read_csv(
                file_path.as_posix(),
                sep="\t",
                header=0,
                encoding="utf-8",
                escapechar="\\",
                quoting=3,
                na_filter=False,
                dtype=dtype,
            )

            # WARNING: instances shorter than the kernel size cannot be convolved.
            min_length = int(self.tokenizer["src"].min_length)
            df['n_frames'] = df[df['n_frames'] > min_length]['n_frames']

            # drop invalid rows
            df = df.replace(r'^\s*$', float("nan"), regex=True)
            df = df.dropna()

            assert "src" in df.columns

            if "trg" not in df.columns:
                self.has_trg = False
                assert self.split == "test"

            if self.has_trg:
                df["trg"] = df["trg"].apply(self.tokenizer["trg"].pre_process)
            return df

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    @property
    def src(self) -> List[str]:
        return self.df["src"]


class StreamDataset(BaseDataset):
    """
    StreamDataset which interacts with stream inputs.
    - called by `translate()` func in `prediction.py`.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "test",
        has_trg: bool = False,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "MT",
        **kwargs,
    ):
        # pylint: disable=unused-argument
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
        )
        # place holder
        self.cache = {}

    def set_item(self, src_line: str, trg_line: str = None) -> None:
        """
        Set input text to the cache.

        :param src_line: (str)
        :param trg_line: (str)
        """
        assert isinstance(src_line, str) and src_line.strip() != "", \
            "The input sentence is empty! Please make sure " \
            "that you are feeding a valid input."

        idx = len(self.cache)
        src_line = self.tokenizer[self.src_lang].pre_process(src_line)

        if self.has_trg:
            trg_line = self.tokenizer[self.trg_lang].pre_process(trg_line)
        self.cache[idx] = (src_line, trg_line)

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        # pylint: disable=unused-argument
        assert idx in self.cache, (idx, self.cache)
        assert lang in [self.src_lang, self.trg_lang]
        if lang == self.trg_lang:
            assert self.has_trg

        line = {}
        src_line, trg_line = self.cache[idx]
        line[self.src_lang] = src_line
        line[self.trg_lang] = trg_line

        item = self.tokenizer[lang](line[lang], is_train=False)
        return item

    def reset_cache(self) -> None:
        self.cache = {}

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(split={self.split}, len={len(self.cache)}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg}, random_subset={self.random_subset})")


class SpeechStreamDataset(SpeechDataset):
    """
    SpeechStreamDataset which interacts with audio file inputs.
    - called by `translate()` func in `prediction.py`.
    """

    def __init__(
        self,
        path: str,
        src_lang: str = "src",
        trg_lang: str = "trg",
        split: int = "test",
        has_trg: bool = False,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "S2T",
        **kwargs,
    ):
        # pylint: disable=unused-argument
        super(TsvDataset, self).__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
        )

        # place holder (empty dataframe)
        try:
            # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
            import pandas as pd
            self.df = pd.DataFrame({'id': [], 'src': [], 'n_frames': []})

            assert isinstance(self.tokenizer["src"], SpeechProcessor)
            self.tokenizer["src"].root_path = Path("")

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def set_item(self, src_line: str, trg_line: str = None) -> None:
        """
        Set input text to the cache.

        :param src_line: (str) absolute path to an audio file
        :param trg_line: (str)
        """
        assert isinstance(src_line, str) and src_line.strip() != "", \
            "The input sentence is empty! Please make sure " \
            "that you are feeding a valid input."

        assert (Path(self.tokenizer["src"].root_path) / src_line).is_file(), \
            f"{src_line} not found. Please provide the abosolute path to a file!"

        min_length = int(self.tokenizer["src"].min_length)  # minimum length
        row = {"id": str(len(self.df)), "src": src_line, "n_frames": min_length}

        if trg_line:
            row["trg"] = self.tokenizer[self.trg_lang].pre_process(trg_line)

        row_df = pd.DataFrame.from_records([row])
        self.df = pd.concat([self.df, row_df], ignore_index=True)

    def reset_cache(self) -> None:
        self.df = pd.DataFrame({'id': [], 'src': [], 'n_frames': []})


class BaseHuggingfaceDataset(BaseDataset):
    """
    Wrapper for Huggingface's dataset object
    cf.) https://huggingface.co/docs/datasets
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        task: str = "MT",
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=kwargs["split"],
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
        )
        # load data
        self.dataset = self.load_data(path, **kwargs)
        self._kwargs = kwargs  # should contain arguments passed to `load_dataset()`

    def load_data(self, path: str, **kwargs) -> Any:
        # pylint: disable=import-outside-toplevel
        try:
            from datasets import config, load_dataset, load_from_disk
            if Path(path, config.DATASET_STATE_JSON_FILENAME).exists():
                return load_from_disk(path)
            return load_dataset(path, **kwargs)

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def sample_random_subset(self, seed: int = 42) -> None:
        super().sample_random_subset(seed)  # check validity

        # resample every epoch: seed += epoch_no
        self.dataset = self.dataset.shuffle(seed=seed).select(range(self.random_subset))

    def reset_random_subset(self) -> None:
        # reload from cache
        self.dataset = self.load_data(self.path, **self._kwargs)

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        # lookup
        line = self.dataset[idx]
        assert lang in line, (line, lang)

        # tokenize
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line[lang], is_train=is_train)
        return item

    def get_list(self,
                 lang: str,
                 postproccessed: bool = False,
                 tokenized: bool = False) -> List[str]:
        if postproccessed:  # pylint: disable=no-else-return
            if f"post_{lang}" not in self.dataset:
                self.dataset = self.dataset.map(
                    lambda item:
                    {f"post_{lang}": self.tokenizer[lang].post_process(item[lang])},
                    desc=f"Postprocessing {lang}...")
            return self.dataset[f"post_{lang}"]
        elif tokenized:
            if f"tok_{lang}" not in self.dataset:
                self.dataset = self.dataset.map(
                    lambda item:
                    {f"tok_{lang}": self.tokenizer[lang](item[lang], is_train=False)},
                    desc=f"Tokenizing {lang}...")
            return self.dataset[f"tok_{lang}"]
        else:
            return self.dataset[lang]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __repr__(self) -> str:
        ret = (f"{self.__class__.__name__}(len={self.__len__()}, "
               f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
               f"has_trg={self.has_trg}, random_subset={self.random_subset}")
        for k, v in self._kwargs.items():
            ret += f", {k}={v}"
        ret += ")"
        return ret


class HuggingfaceDataset(BaseHuggingfaceDataset):
    """
    Wrapper for Huggingface's `datasets.features.Translation` class
    cf.) https://github.com/huggingface/datasets/blob/master/src/datasets/features/translation.py
    """  # noqa

    def load_data(self, path: str, **kwargs) -> Any:
        dataset = super().load_data(path=path, **kwargs)

        # rename columns
        if "translation" in dataset.features:
            # check language pair
            lang_pair = dataset.features["translation"].languages
            assert self.src_lang in lang_pair, (self.src_lang, lang_pair)

            # rename columns
            columns = {f"translation.{self.src_lang}": self.src_lang}
            if self.has_trg:
                assert self.trg_lang in lang_pair, (self.trg_lang, lang_pair)
                columns[f"translation.{self.trg_lang}"] = self.trg_lang

            # flatten
            dataset = dataset.flatten()

        elif f"{self.src_lang}_sentence" in dataset.features:
            # rename columns
            columns = {f"{self.src_lang}_sentence": self.src_lang}
            if self.has_trg:
                assert f"{self.trg_lang}_sentence" in dataset.features
                columns[f"{self.trg_lang}_sentence"] = self.trg_lang

        else:
            pass
            # TODO: support other field names
        dataset = dataset.rename_columns(columns)

        # preprocess (lowercase, pretokenize, etc.)
        def _pre_process(item):
            sl = self.src_lang
            tl = self.trg_lang
            ret = {sl: self.tokenizer[sl].pre_process(item[sl])}
            if self.has_trg:
                ret[tl] = self.tokenizer[tl].pre_process(item[tl])
            return ret

        def _drop_nan(item):
            sl = self.src_lang
            tl = self.trg_lang
            is_src_valid = item[sl] is not None and len(item[sl]) > 0
            if self.has_trg:
                is_trg_valid = item[tl] is not None and len(item[tl]) > 0
                return is_src_valid and is_trg_valid
            return is_src_valid

        dataset = dataset.filter(_drop_nan, desc="Dropping NaN...")
        return dataset.map(_pre_process, desc="Preprocessing...")


def build_dataset(
    dataset_type: str,
    path: str,
    src_lang: str,
    trg_lang: str,
    split: str,
    tokenizer: Dict = None,
    sequence_encoder: Dict = None,
    random_subset: int = -1,
    task: str = "MT",
    **kwargs,
):
    """
    Builds a dataset.

    :param dataset_type: (str) one of {`plain`, `tsv`, `stream`, `huggingface`}
    :param path: (str) either a local file name or
        dataset name to download from remote
    :param src_lang: (str) language code for source
    :param trg_lang: (str) language code for target
    :param split: (str) one of {`train`, `dev`, `test`}
    :param tokenizer: tokenizer objects for both source and target
    :param sequence_encoder: encoding functions for both source and target
    :param random_subset: (int) number of random subset; -1 means no subsampling
    :param task: (str)
    :return: loaded Dataset
    """
    dataset = None
    has_trg = True  # by default, we expect src-trg pairs

    if dataset_type == "plain":
        if not Path(path).with_suffix(f"{Path(path).suffix}.{trg_lang}").is_file():
            # no target is given -> create dataset from src only
            has_trg = False
        dataset = PlaintextDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
            **kwargs,
        )
    elif dataset_type == "tsv":
        dataset = TsvDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
            **kwargs,
        )
    elif dataset_type == "speech":
        dataset = SpeechDataset(
            path=path,
            src_lang="src",
            trg_lang="trg",
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
            **kwargs,
        )
    elif dataset_type == "stream":
        dataset = StreamDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="test",
            has_trg=False,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=-1,
            task=task,
            **kwargs,
        )
    elif dataset_type == "speech_stream":
        dataset = SpeechStreamDataset(
            path=None,
            src_lang="src",
            trg_lang="trg",
            split="test",
            has_trg=False,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=-1,
            task=task,
            **kwargs,
        )
    elif dataset_type == "huggingface":
        # "split" should be specified in kwargs
        if "split" not in kwargs:
            kwargs["split"] = "validation" if split == "dev" else split
        dataset = HuggingfaceDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            task=task,
            **kwargs,
        )
    else:
        ConfigurationError(f"{dataset_type}: Unknown dataset type.")
    return dataset


class SentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of instances.
    An instance longer than dataset.max_len will be filtered out.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if its size
        would be less than `batch_size`
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        d = self.sampler.data_source
        for idx in self.sampler:
            src, trg = d[idx]  # pylint: disable=unused-variable
            if src is not None:  # otherwise drop instance
                batch.append(idx)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class TokenBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of tokens
    (incl. padding). An instance longer than dataset.max_len or shorter than
    dataset.min_len will be filtered out.
    * no bucketing implemented

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if
            its size would be less than `batch_size`
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        max_tokens = 0
        d = self.sampler.data_source
        for idx in self.sampler:
            src, trg = d[idx]  # call __getitem__()
            if src is not None:  # otherwise drop instance
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                n_tokens = 0 if src_len == 0 else max(src_len + 1, trg_len + 2)
                batch.append(idx)
                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError
