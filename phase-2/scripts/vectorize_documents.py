import argparse
from typing import cast

import torch
from gensim.models.doc2vec import Doc2Vec

from utils import CorpusUtils


BASE_PATH = "./0327-1503/"
PATH = f"{BASE_PATH}example-data.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-path",
        help="Path to load pre-trained embedding model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the tokenized document file for generating the corpus file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="path to save the corpus file",
        type=str,
    )
    args = parser.parse_args()

    path = args.path if args.path else PATH

    output_path = args.output_pathif if args.output_path else f"{path}.pt"

    model: Doc2Vec = cast(Doc2Vec, Doc2Vec.load(args.model_path))
    data_to_write: list[tuple[str, torch.Tensor]] = []
    for label, *words in CorpusUtils.read_data(path):
        vector: torch.Tensor = torch.from_numpy(model.infer_vector(words))
        data_to_write.append((label, vector))

    torch.save(data_to_write, output_path)


if __name__ == "__main__":
    main()
