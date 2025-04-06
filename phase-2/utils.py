import csv
import logging
import os
from functools import singledispatchmethod
from typing import Generator, Union, Iterable, cast

import torch
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split


class CorpusUtils:
    @staticmethod
    def read_data(
        path="./example-data.csv",
        to_tagged_document=False,
        tokens_only=False,
        vectorized=False,
    ) -> Union[
        Generator[list[str], None, None],
        Generator[TaggedDocument, None, None],
        Generator[tuple[str, torch.Tensor], None, None],
    ]:
        if vectorized:
            return CorpusUtils._read_vectorized_data(path)
        return CorpusUtils._read_raw_data(path, to_tagged_document, tokens_only)

    @staticmethod
    def _read_raw_data(
        path,
        to_tagged_document=False,
        tokens_only=False,
    ) -> Union[
        Generator[list[str], None, None],
        Generator[TaggedDocument, None, None],
    ]:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.split(",")
                if len(parts) <= 1:
                    continue

                if to_tagged_document:
                    yield TaggedDocument(words=parts[1:], tags=[parts[0]])

                if tokens_only:
                    yield parts[1:]
                else:
                    yield parts

    @staticmethod
    def _read_vectorized_data(
        path="./example-data.csv.pt",
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for label, *feature in torch.load(path):
            yield label, *feature

    @staticmethod
    def to_tagged_documents(
        lines: Iterable[list[str]],
    ) -> Generator[TaggedDocument, None, None]:
        for parts in lines:
            if len(parts) <= 1:
                continue
            yield TaggedDocument(words=parts[1:], tags=[parts[0]])

    @staticmethod
    def spllit_data_from_file(
        path="./example-data.csv",
        write_as_file=False,
        vectorized=False,
    ) -> tuple[
        Union[
            list[list[str]],
            list[tuple[str, torch.Tensor]],
        ],
        Union[
            list[list[str]],
            list[tuple[str, torch.Tensor]],
        ],
    ]:
        tokenized_titles = list(
            CorpusUtils.read_data(path, to_tagged_document=False, vectorized=vectorized)
        )
        datasets = tuple(
            train_test_split(
                tokenized_titles,
                test_size=0.2,
                shuffle=True,
                random_state=42,
            )
        )

        if not write_as_file:
            return tuple(datasets)

        for i, dataset in enumerate(datasets):
            train_or_test = "train" if i % 2 == 0 else "test"
            new_file_path = f"{path}.{train_or_test}"

            with open(new_file_path, "w") as file:
                writer = csv.writer(file)
                for parts in dataset:
                    writer.writerow(parts)

        return tuple(datasets)

    @staticmethod
    def read_splited_data(
        path="./example-data.csv",
    ):
        SPLITED_DATASET_PATH = (f"{path}.train", f"{path}.test")

        datasets = []
        for splited_dataset_path in SPLITED_DATASET_PATH:
            if not os.path.exists(splited_dataset_path):
                continue

            splited_dataset = list(
                CorpusUtils.read_data(path, to_tagged_document=False)
            )
            datasets.append(splited_dataset)
        return datasets

    @staticmethod
    def vectorize_corpus(
        data: list[list[str]], embedding_model: Doc2Vec
    ) -> list[tuple[str, torch.Tensor]]:
        return [
            (label, torch.from_numpy(embedding_model.infer_vector(feature)))
            for label, *feature in data
        ]

    @staticmethod
    def vectorize_tagged_documents(
        data: list[TaggedDocument], embedding_model: Doc2Vec
    ) -> list[tuple[str, torch.Tensor]]:
        return [
            (
                tagged_document.tags[0],
                torch.from_numpy(embedding_model.infer_vector(tagged_document.words)),
            )
            for tagged_document in data
        ]

    @staticmethod
    def vectorize(data: list[str], embedding_model: Doc2Vec) -> torch.Tensor:
        return torch.from_numpy(embedding_model.infer_vector(data))


class ModelUtils:
    @singledispatchmethod
    @staticmethod
    def setup_model_configuration(config_or_path) -> Doc2Vec:
        raise NotImplementedError

    @setup_model_configuration.register
    @staticmethod
    def _(config: dict) -> Doc2Vec:
        logging.info(f"Model Configuration - {config}")
        return Doc2Vec(**config)

    @setup_model_configuration.register
    @staticmethod
    def _(path: str) -> Doc2Vec:
        logging.info(f"Model Configuration - read model from {path}")
        return cast(Doc2Vec, Doc2Vec.load(path))
