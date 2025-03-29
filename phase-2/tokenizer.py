from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import pandas as pd
import csv
import torch
from datetime import datetime
import os

CONJUNCTIONS = ["Caa", "Cab", "Cba", "Cbb", "DE"]
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


def is_conjunctions(word_pos: str) -> bool:
    return word_pos in CONJUNCTIONS


def is_prepositions(word_pos: str) -> bool:
    return word_pos == "P"


def is_punctuation(word_pos: str) -> bool:
    return word_pos in PUNCTUATION


def is_particle(word_pos: str) -> bool:
    return word_pos == "T"


def is_N_or_A_or_V_or_FW(word_pos: str) -> bool:
    return (
        word_pos.startswith("N")
        or word_pos.startswith("V")
        or word_pos == "A"
        or word_pos == "FW"
    )


def tokenize(title_ws, title_pos, show_pos=False) -> list[str]:
    return [
        f"{word_ws.strip()}{f'({word_pos})' if show_pos else ''}"
        for word_ws, word_pos in zip(title_ws, title_pos)
        if not (
            is_conjunctions(word_pos)
            or is_prepositions(word_pos)
            or is_punctuation(word_pos)
            or is_particle(word_pos)
        )
    ]


def tokenize_navfw(title_ws, title_pos, show_pos=False) -> list[str]:
    return [
        f"{word_ws.strip()}{f'({word_pos})' if show_pos else ''}"
        for word_ws, word_pos in zip(title_ws, title_pos)
        if is_N_or_A_or_V_or_FW(word_pos)
    ]


def main():
    device = 0 if torch.cuda.is_available() else -1
    ws_driver = CkipWordSegmenter(model="bert-base", device=device)
    pos_driver = CkipPosTagger(model="bert-base", device=device)

    df = pd.read_csv("./labeled-title.csv")

    current_datetime = datetime.now().strftime("%m%d-%H%M")
    output_dir = f"./{current_datetime}/"
    os.makedirs(output_dir, exist_ok=True)

    rows_per_chunk = 10000
    batch_size = 64
    for i in range(0, df.shape[0], rows_per_chunk):
        start_idx = i
        end_idx = i + rows_per_chunk - 1  # loc contains end index
        titles = df.loc[start_idx:end_idx, "title"]
        labels = df.loc[start_idx:end_idx, "board_name"]

        ws = ws_driver(titles, batch_size=batch_size)
        pos = pos_driver(ws, batch_size=batch_size)

        with (
            open(f"{output_dir}tokenized-title", "a") as file1,
            open(f"{output_dir}tokenized-title-with-pos", "a") as file2,
            open(f"{output_dir}tokenized-title-only-NAVFW", "a") as file3,
            open(f"{output_dir}tokenized-title-with-pos-only-NAVFW", "a") as file4,
        ):
            writer1 = csv.writer(file1)
            writer2 = csv.writer(file2)
            writer3 = csv.writer(file3)
            writer4 = csv.writer(file4)

            for label, _, title_ws, title_pos in zip(labels, titles, ws, pos):
                results = tokenize(title_ws, title_pos)
                writer1.writerow([label, *results])

                results = tokenize(title_ws, title_pos, show_pos=True)
                writer2.writerow([label, *results])

                results = tokenize_navfw(title_ws, title_pos)
                writer3.writerow([label, *results])

                results = tokenize_navfw(title_ws, title_pos, show_pos=True)
                writer4.writerow([label, *results])

            print(f"Round {i} Write to Files: Finished")


if __name__ == "__main__":
    main()
