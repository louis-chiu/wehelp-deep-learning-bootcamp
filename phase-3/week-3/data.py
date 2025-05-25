from ckip_transformers.nlp import CkipWordSegmenter

import json
import os
import torch
from torch.utils.data import Dataset

BASE_PATH = "./data/"  # "phase-3/week-3/data/"


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def load_paragraphs_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [p for chapter in data for p in chapter["paragraphs"]]


def write_output(ws_list, output_dir):
    txt_path = os.path.join(output_dir, "corpus.txt")

    with (
        open(txt_path, "w", encoding="utf-8") as txt_f,
    ):
        for ws in ws_list:
            tokens = [w for w in zip(ws)]
            txt_f.write(" ".join(tokens) + "\n")


class CorpusDataset(Dataset):
    def __init__(self, seq_len=35, path=f"{BASE_PATH}data.json"):
        super().__init__()
        self.dictionary = Dictionary()
        self.path = path
        self.data = self.tokenize()
        self.seq_len = seq_len

    def tokenize(self):
        device = 0 if torch.cuda.is_available() else -1
        ws_driver = CkipWordSegmenter(model="bert-base", device=device)

        paragraphs = load_paragraphs_from_json(self.path)
        batch_size = 32

        ws_list = ws_driver(paragraphs, batch_size=batch_size)

        # Add words to the dictionary
        for words in ws_list:
            words += ["<eos>"]
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        idss = []
        for words in ws_list:
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

        return ids

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        seq_start = idx * self.seq_len
        seq_end = seq_start + self.seq_len
        x = self.data[seq_start:seq_end]
        y = self.data[seq_start + 1 : seq_end + 1]
        return x, y
