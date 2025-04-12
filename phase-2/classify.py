import logging
from utils import CorpusUtils, ModelUtils
from datetime import datetime
from typing import cast, Union

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt


BASE_PATH = ""
PATH = f"{BASE_PATH}example-data.csv.pt"
# BASE_PATH = "./0327-1503/"
# PATH = f"{BASE_PATH}tokenized-title.pt"
VECTORIZED = True  # indicates if corpus has been vectorized already
EMBEDDING_MODEL_PATH = f"{BASE_PATH}0402-0002-75-86.model"
EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")
LEARNING_CURVE_PATH = f"{BASE_PATH}{EXECUTE_AT}-learning_curve.png"


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
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 9),
        )

    def forward(self, x):
        return self.model(x)


class Task:
    @staticmethod
    def run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Training Using {device}")
        base_lr = 0.015
        batch_size = 8  # 128
        learning_rate = base_lr * (batch_size**0.5)
        epochs = 10  # 500
        embedding_model = ModelUtils.setup_model_configuration(EMBEDDING_MODEL_PATH)
        logging.info(
            f"Model Configuration {batch_size=}, {learning_rate=}, {epochs=}, {PATH=}"
        )

        logging.info("Loadding Dataset...")

        dataset = PTTDataset(embedding_model)
        train_size = int(0.8 * len(dataset))
        validate_size = int((len(dataset) - train_size) / 2)
        test_size = len(dataset) - train_size - validate_size
        train_dataset, rest_dataset = random_split(
            dataset,
            [train_size, validate_size + test_size],
            generator=torch.Generator().manual_seed(42),
        )
        validate_dataset, test_dataset = random_split(
            rest_dataset,
            [validate_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        val_loader = DataLoader(
            validate_dataset,
            batch_size,
            shuffle=False,
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

        train_losses = []
        train_first_accuracies = []
        train_second_accuracies = []

        val_losses = []
        val_first_accuracies = []
        val_second_accuracies = []
        logging.info("Model Training ...")
        for epoch in range(epochs):
            # Training
            loop = tqdm(enumerate(train_loader), total=len(train_loader))

            train_first_match = 0
            train_second_match = 0
            train_total_loss = 0

            model.train()
            for i, (labels, features) in loop:
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                optimizer.zero_grad()

                prediction = model(features)
                loss = loss_fn(prediction, targets)
                loss.backward()
                optimizer.step()

                for j, label in enumerate(labels):
                    flatted_prediction = prediction[j].unsqueeze(0)
                    target_label_index = TAG_MAPPING.get(label)
                    if Task.is_prediction_correct(
                        flatted_prediction, target_label_index
                    ):
                        train_first_match += 1
                    if Task.is_prediction_correct(
                        flatted_prediction, target_label_index, 2
                    ):
                        train_second_match += 1
                train_total_loss += loss.item()

                loop.set_description(
                    f"Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():>10.5}"
                )

                if i == len(train_loader) - 1:  # only write the last data into log file
                    logging.info(
                        f"Train - Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():>10.5}"
                    )

            train_total_loss /= len(train_dataset)
            train_first_acc = train_first_match / len(train_dataset)
            train_second_acc = train_second_match / len(train_dataset)

            train_losses.append(train_total_loss)
            train_first_accuracies.append(train_first_acc)
            train_second_accuracies.append(train_second_acc)

            # Validation
            val_first_match = 0
            val_second_match = 0
            val_total_loss = 0

            model.eval()
            with torch.no_grad():
                for labels, features in val_loader:
                    features = features.to(device)
                    targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                    prediction = model(features)
                    loss = loss_fn(prediction, targets)

                    for i, label in enumerate(labels):
                        flatted_prediction = prediction[i].unsqueeze(0)
                        target_label_index = TAG_MAPPING.get(label)
                        if Task.is_prediction_correct(
                            flatted_prediction, target_label_index
                        ):
                            val_first_match += 1
                        if Task.is_prediction_correct(
                            flatted_prediction, target_label_index, 2
                        ):
                            val_second_match += 1
                    val_total_loss += loss.item()

                    logging.info(
                        f"Validation - Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():>10.5}"
                    )

                val_total_loss /= len(validate_dataset)
                val_first_acc = val_first_match / len(validate_dataset)
                val_second_acc = val_second_match / len(validate_dataset)

                val_losses.append(val_total_loss)
                val_first_accuracies.append(val_first_acc)
                val_second_accuracies.append(val_second_acc)
        Task.plot_learning_curves(
            train_losses,
            val_losses,
            train_first_accuracies,
            val_first_accuracies,
            train_second_accuracies,
            val_second_accuracies,
        )
        # Evaluate after training
        model.eval()
        test_first_match = 0
        test_second_match = 0
        total_samples = 0
        logging.info("Model Evaluating After Training ...")
        with torch.no_grad():
            for _, (labels, features) in enumerate(test_loader):
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                prediction = model(features)
                loss = loss_fn(prediction, targets)

                for i, label in enumerate(labels):
                    flatted_prediction = prediction[i].unsqueeze(0)
                    target_label_index = TAG_MAPPING.get(label)
                    if Task.is_prediction_correct(
                        flatted_prediction, target_label_index
                    ):
                        test_first_match += 1
                    if Task.is_prediction_correct(
                        flatted_prediction, target_label_index, 2
                    ):
                        test_second_match += 1
                total_samples += len(labels)

        test_first_accuracy = test_first_match / total_samples
        test_second_accuracy = test_second_match / total_samples
        logging.info(f"Accuracy after traning: {test_first_accuracy}")

        torch.save(
            model,
            f"{BASE_PATH}{EXECUTE_AT}-first-{int(test_first_accuracy * 100)}-second-{int(test_second_accuracy * 100)}.classify.model",
        )

    @staticmethod
    def is_prediction_correct(
        output: torch.Tensor, target_index: Union[int, None], nth_matched: int = 1
    ) -> bool:
        return bool((output.topk(nth_matched).indices == target_index).any().item())

    @staticmethod
    def one_hot_encoding(labels, tag_mapping, device):
        targets = torch.zeros(len(labels), len(tag_mapping), device=device)
        batch_indices = torch.arange(len(labels), device=device)
        tag_indices = torch.tensor(
            [tag_mapping.get(label) for label in labels], device=device
        )
        targets[batch_indices, tag_indices] = 1
        return targets

    @staticmethod
    def plot_learning_curves(
        train_losses,
        val_losses,
        train_first_match_accuracies,
        val_first_match_accuracies,
        train_second_match_accuracies,
        val_second_match_accuracies,
        save_path=LEARNING_CURVE_PATH,
    ):
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_first_match_accuracies, label="Training 1st-Matched Accuracy")
        plt.plot(val_first_match_accuracies, label="Validation 1st-Matched Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_second_match_accuracies, label="Training 2nd-Matched Accuracy")
        plt.plot(val_second_match_accuracies, label="Validation 2nd-Matched Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}_learning_curves.png")
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=f"{BASE_PATH}classify-{EXECUTE_AT}.log",
    )
    Task.run()
