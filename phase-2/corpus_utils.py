from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import csv
import os


def read_titles(path="./example-data.csv", to_tagged_document=False, tokens_only=False):
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


def to_tagged_documents(lines: list[list[str]]):
    for parts in lines:
        if len(parts) <= 1:
            continue
        yield TaggedDocument(words=parts[1:], tags=[parts[0]])


def spllit_data(
    path="./example-data.csv",
    write_as_file=False,
    vectorized=False,
):
    SPLITED_DATASET_PATH = (f"{path}.train", f"{path}.test")

    datasets = []
    for splited_dataset_path in SPLITED_DATASET_PATH:
        if not os.path.exists(splited_dataset_path):
            continue

        splited_dataset = list(read_titles(path, to_tagged_document=False))
        datasets.append(splited_dataset)

    if not datasets:
        tokenized_titles = list(read_titles(path, to_tagged_document=False))
        datasets = train_test_split(
            tokenized_titles,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )

    if not write_as_file:
        return datasets
    
    for i, dataset in enumerate(datasets):
        train_or_test = "train" if i % 2 == 0 else "test"
        new_file_path = f"{path}.{train_or_test}"

        with open(new_file_path, "w") as file:
            writer = csv.writer(file)
            for parts in dataset:
                writer.writerow(parts)
    
    return datasets
