import torch
import torch.nn as nn
import math
from collections import Counter

import torch.nn.functional as F
from data import CorpusDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_PATH = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers
        )

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


def train():
    batch_size = 64
    lr = 0.001
    epochs = 50
    emsize = 128  # 嵌入向量維度
    nhead = 2  # 注意力頭數
    nhid = 128  # FFN維度
    nlayers = 2  # encoder層數
    dropout = 0.3

    dataset = CorpusDataset()
    train_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    ntokens = len(dataset.dictionary)

    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        loop = tqdm(
            train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}"
        )
        for index, (inputs, targets) in enumerate(loop):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            output = output.view(-1, output.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

    save_checkpoint(
        model,
        f"{BASE_PATH}model/model.pt",
        dataset.dictionary,
        top20_freq_initial_tokens(dataset),
    )


def save_checkpoint(model, save_path, vocab, initial_tokens_freq):
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": {
            "vocab_size": model.input_emb.num_embeddings,
            "ninp": model.ninp,
            "nhead": model.nhead,
            "nhid": model.encoder.layers[0].linear1.in_features,
            "nlayers": len(model.encoder.layers),
            "dropout": model.pos_encoder.dropout.p,
        },
        "initial_tokens_freq": initial_tokens_freq,
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def top20_freq_initial_tokens(dataset):
    token_counter = Counter()

    data = dataset.data
    eos_idx = dataset.dictionary.word2idx["<eos>"]

    mark_as_end = True
    for token in data:
        token = token.item()
        if mark_as_end:
            mark_as_end = False
            token_counter[token] += 1

        if token == eos_idx:
            mark_as_end = True

    most_common_tokens = dict(token_counter.most_common(20))

    return most_common_tokens


if __name__ == "__main__":
    train()
