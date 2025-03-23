from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import pandas as pd
import csv
import torch

CONJUNCTIONS = ["Caa", "Cab", "Cba", "Cbb"]
PUNCTUATION = [
    "COLONCATEGORY",
    "COMMACATEGORY",
    "DASHCATEGORY",
    "DOTCATEGORY",
    "ETCCATEGORY",
    "EXCLAMATIONCATEGORY",
    "PARENTHESISCATEGORY",
    "PAUSECATEGORY",
    "PERIODCATEGORY",
    "QUESTIONCATEGORY",
    "SEMICOLONCATEGORY",
    "SPCHANGECATEGORY",
    "WHITESPACE",
]


def is_conjunctions(word_pos) -> bool:
    return word_pos in CONJUNCTIONS


def is_prepositions(word_pos) -> bool:
    return word_pos == "P"


def is_punctuation(word_pos) -> bool:
    return word_pos in PUNCTUATION


def tokenizer(title_ws, title_pos) -> list[str]:
    return [
        f"{word_ws}"
        for word_ws, word_pos in zip(title_ws, title_pos)
        if not (
            is_conjunctions(word_pos)
            or is_prepositions(word_pos)
            or is_punctuation(word_pos)
        )
    ]


def main():
    device = 0 if torch.cuda.is_available() else -1
    ws_driver = CkipWordSegmenter(model="bert-base", device=device)
    pos_driver = CkipPosTagger(model="bert-base", device=device)
    ner_driver = CkipNerChunker(model="bert-base", device=device)

    df = pd.read_csv("./example-data.csv")

    rows_per_chunk = 1000
    batch_size = 64
    for i in range(0, df.shape[0], rows_per_chunk):
        start_idx = i
        end_idx = i + rows_per_chunk - 1  # loc contains end index
        titles = df.loc[start_idx:end_idx, "title"]
        labels = df.loc[start_idx:end_idx, "board_name"]

        ws = ws_driver(titles, batch_size=batch_size)
        pos = pos_driver(ws, batch_size=batch_size)
        ner = ner_driver(titles, batch_size=batch_size)

        with open("./tokenizer-title.csv", "a") as file:
            writer = csv.writer(file)
            for label, _, title_ws, title_pos, _ in zip(labels, titles, ws, pos, ner):
                results = tokenizer(title_ws, title_pos)
                writer.writerow([label, *results])


if __name__ == "__main__":
    main()
