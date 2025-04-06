import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import logging
from utils import CorpusUtils, ModelUtils
from datetime import datetime
from typing import cast


BASE_PATH = ""
PATH = f"{BASE_PATH}example-data.csv.pt"
# BASE_PATH = "./0327-1503/"
# PATH = f"{BASE_PATH}tokenized-title.pt"
VECTORIZED = True  # indicates if corpus has been vectorized already
EMBEDDING_MODEL_PATH = f"{BASE_PATH}0402-0002-75-86.model"
EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")


class PTTDataset(Dataset):
    def __init__(self, embedding_model) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.data = [
            cast(tuple[str, torch.Tensor], line)
            if VECTORIZED
            else CorpusUtils.vectorize(cast(list[str], line), embedding_model)
            for line in CorpusUtils.read_data(PATH, vectorized=VECTORIZED)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[str, torch.Tensor]:
        return self.data[index]


class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
        )

    def forward(self, x):
        return self.model(x)


class Task:
    @staticmethod
    def run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Training Using {device}")
        base_lr = 0.015
        batch_size = 1
        learning_rate = base_lr * (batch_size**0.5)
        epochs = 100
        embedding_model = ModelUtils.setup_model_configuration(EMBEDDING_MODEL_PATH)
        logging.info(f"Model Configuration {batch_size=}, {learning_rate=}, {epochs=}")

        logging.info("Loadding Dataset...")

        dataset = PTTDataset(embedding_model)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        model = ClassificationNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        TAG_MAPPING: dict[str, int] = {
            tag: index for index, tag in enumerate(embedding_model.dv.index_to_key)
        }  # type: ignore

        # Evaluate before training
        model.eval()
        correct_times = 0
        total_samples = 0
        logging.info("Model Evaluating Before Training ...")
        with torch.no_grad():
            for _, (labels, features) in enumerate(test_loader):
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                outputs = model(features)
                loss = loss_fn(outputs, targets)

                for i, label in enumerate(labels):
                    if Task.is_prediction_correct(
                        outputs[i].unsqueeze(0), TAG_MAPPING.get(label)
                    ):
                        correct_times += 1
                total_samples += len(labels)

        accuracy = correct_times / total_samples
        logging.info(f"Accuracy before traning: {accuracy}")

        # Training
        model.train()
        logging.info("Model Training ...")
        for epoch in range(epochs):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (labels, features) in loop:
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                optimizer.zero_grad()

                prediction = model(features)
                loss = loss_fn(prediction, targets)
                loss.backward()
                optimizer.step()

                loop.set_description(
                    f"Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():<10.5}"
                )

                if i == len(train_loader) - 1:
                    logging.info(
                        f"Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():<10.5}"
                    )

        # Evaluate after training
        model.eval()
        correct_times = 0
        total_samples = 0
        logging.info("Model Evaluating After Training ...")
        with torch.no_grad():
            for _, (labels, features) in enumerate(test_loader):
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                outputs = model(features)
                loss = loss_fn(outputs, targets)

                for i, label in enumerate(labels):
                    if Task.is_prediction_correct(
                        outputs[i].unsqueeze(0), TAG_MAPPING.get(label)
                    ):
                        correct_times += 1
                total_samples += len(labels)

        accuracy = correct_times / total_samples
        logging.info(f"Accuracy after traning: {accuracy}")

        torch.save(
            model.state_dict(),
            f"{BASE_PATH}{EXECUTE_AT}-{int(accuracy * 100)}.classify.model",
        )

    @staticmethod
    def is_prediction_correct(output, target_index) -> bool:
        return output.argmax().item() == target_index

    @staticmethod
    def one_hot_encoding(labels, tag_mapping, device):
        targets = torch.zeros(len(labels), len(tag_mapping), device=device)
        batch_indices = torch.arange(len(labels), device=device)
        tag_indices = torch.tensor(
            [tag_mapping.get(label) for label in labels], device=device
        )
        targets[batch_indices, tag_indices] = 1
        return targets


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=f"{BASE_PATH}classify-{EXECUTE_AT}.log",
    )
    Task.run()
