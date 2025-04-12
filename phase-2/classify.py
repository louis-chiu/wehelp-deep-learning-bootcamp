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
# PATH = f"{BASE_PATH}tokenized-title-only-NAVFW.pt"
VECTORIZED = True  # indicates if corpus has been vectorized already
EMBEDDING_MODEL_PATH = f"{BASE_PATH}0402-0002-75-86.model"
EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")
LEARNING_CURVE_PATH = f"{BASE_PATH}{EXECUTE_AT}-learning-curve.png"


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
        batch_size = 1
        learning_rate = base_lr * (batch_size**0.5)
        epochs = 15
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

        validation_epochs = Task.get_validation_epochs(epochs)

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
            train_total_samples = 0

            model.train()
            for i, (labels, features) in loop:
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                optimizer.zero_grad()

                prediction = model(features)
                loss = loss_fn(prediction, targets)
                loss.backward()
                optimizer.step()

                first_match, second_match = Task.calculate_top_k_accuracy(
                    prediction, labels, TAG_MAPPING, device
                )

                loop.set_description(
                    f"Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():>10.5}"
                )

                if epoch in validation_epochs:
                    train_first_match += first_match
                    train_second_match += second_match
                    train_total_samples += len(labels)
                    train_total_loss += loss.item() * len(labels)
            logging.info(
                f"Train - Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {train_total_loss:>10.5}"
            )

            if epoch in validation_epochs:
                train_total_loss /= train_total_samples
                train_first_acc = train_first_match / train_total_samples
                train_second_acc = train_second_match / train_total_samples

                train_losses.append(train_total_loss)
                train_first_accuracies.append(train_first_acc)
                train_second_accuracies.append(train_second_acc)

            # Validation
            if epoch in validation_epochs:
                val_first_match = 0
                val_second_match = 0
                val_total_loss = 0
                val_total_samples = 0

                model.eval()
                with torch.no_grad():
                    for labels, features in val_loader:
                        features = features.to(device)
                        targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                        prediction = model(features)
                        loss = loss_fn(prediction, targets)

                        first_match, second_match = Task.calculate_top_k_accuracy(
                            prediction, labels, TAG_MAPPING, device
                        )
                        val_first_match += first_match
                        val_second_match += second_match
                        val_total_samples += len(labels)
                        val_total_loss += loss.item() * len(labels)

                    val_total_loss /= val_total_samples
                    val_first_acc = val_first_match / val_total_samples
                    val_second_acc = val_second_match / val_total_samples
                    logging.info(
                        f"Validation - Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {val_total_loss:>10.5}"
                    )
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
        test_total_samples = 0
        logging.info("Model Evaluating After Training ...")
        with torch.no_grad():
            for labels, features in test_loader:
                features = features.to(device)
                targets = Task.one_hot_encoding(labels, TAG_MAPPING, device)

                prediction = model(features)
                loss = loss_fn(prediction, targets)
                first_match, second_match = Task.calculate_top_k_accuracy(
                    prediction, labels, TAG_MAPPING, device
                )

                test_first_match += first_match
                test_second_match += second_match
                test_total_samples += len(labels)

        test_first_accuracy = test_first_match / test_total_samples
        test_second_accuracy = test_second_match / test_total_samples
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
    def calculate_top_k_accuracy(
        prediction: torch.Tensor,
        labels: list[str],
        tag_mapping: dict[str, int],
        device: torch.device,
    ) -> tuple:
        top1_preds = prediction.topk(1).indices
        top2_preds = prediction.topk(2).indices
        target_indices = torch.tensor(
            [tag_mapping.get(label) for label in labels], device=device
        ).view(-1, 1)

        first_match = (top1_preds == target_indices).sum().item()
        second_match = (top2_preds == target_indices).any(dim=1).sum().item()
        return first_match, second_match

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
    def get_validation_epochs(epochs: int) -> list[int]:
        if epochs <= 100:
            validation_epochs = list(range(epochs))
        else:
            step = (
                epochs // 99
            )  # 99 points + the final epoch = 100 total validation points
            validation_epochs = list(range(0, epochs, step))

        if epochs - 1 not in validation_epochs:
            validation_epochs.append(epochs - 1)
        return validation_epochs

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
        plt.plot(train_first_match_accuracies, label="Training Accuracy")
        plt.plot(val_first_match_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("1st-Matched Accuracy Curves")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_second_match_accuracies, label="Training Accuracy")
        plt.plot(val_second_match_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("2nd-Matched Accuracy Curves")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}")
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=f"{BASE_PATH}classify-{EXECUTE_AT}.log",
    )
    Task.run()
