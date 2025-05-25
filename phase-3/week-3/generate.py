import torch
from main import TransformerModel
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model/model.pt", "rb") as f:
    checkpoint = torch.load(f, map_location=device, weights_only=False)
    # 從checkpoint重建模型

    model = TransformerModel(
        checkpoint["config"]["vocab_size"],
        checkpoint["config"]["ninp"],
        checkpoint["config"]["nhead"],
        checkpoint["config"]["nhid"],
        checkpoint["config"]["nlayers"],
        checkpoint["config"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

model.eval()

dictionary = checkpoint["vocab"]
initial_tokens_freq = checkpoint["initial_tokens_freq"]


initial_token_keys = list(initial_tokens_freq.keys())
for i in range(1, 6):
    random_word = random.randint(0, len(initial_token_keys) - 1)
    word_idx = initial_token_keys[random_word]
    input = torch.tensor([[word_idx]], dtype=torch.long).to(device)

    first_word = dictionary.idx2word[word_idx]

    with torch.no_grad():  # no tracking history
        temperature = 0.8
        result = [first_word]
        word = ""
        while word != "<eos>":
            result.append(word)

            output = model(input, False)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)

            word = dictionary.idx2word[word_idx]

        print(i, "".join(result))
