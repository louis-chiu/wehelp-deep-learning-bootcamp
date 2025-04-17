import logging
from utils import CorpusUtils, ModelUtils
from datetime import datetime
from typing import cast, Union

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# BASE_PATH = ".example-data-output/"
# PATH = f"{BASE_PATH}example-data.csv.pt"
BASE_PATH = "./0327-1503/"
PATH = f"{BASE_PATH}tokenized-title.pt"
# PATH = f"{BASE_PATH}tokenized-title-only-NAVFW.pt"
VECTORIZED = True  # indicates if corpus has been vectorized already

EMBEDDING_MODEL_PATH = f"{BASE_PATH}0402-0002-75-86.model"
EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")
LEARNING_CURVE_PATH = f"{BASE_PATH}{EXECUTE_AT}-learning-curve.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 500


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
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 9),
        )

    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    def __init__(
        self, patience=15, delta=0.001, save_path="best_model.pt", verbose=True
    ):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)  # 只是要暫存
        if self.verbose:
            print(f"Validation loss improved. Model saved to {self.save_path}")


class Classifier:
    @staticmethod
    def run():
        logging.info(f"Training Using {DEVICE}")

        logging.info(
            f"Model Configuration {BATCH_SIZE=}, {LEARNING_RATE=}, {EPOCHS=}, {PATH=}, {EMBEDDING_MODEL_PATH=}"
        )
        embedding_model = ModelUtils.setup_model_configuration(EMBEDDING_MODEL_PATH)

        logging.info("Loadding Dataset...")
        dataset = PTTDataset(embedding_model)
        train_size = int(0.8 * len(dataset))
        validate_size = int((len(dataset) - train_size) / 2)
        test_size = len(dataset) - train_size - validate_size

        train_dataset, validate_dataset, test_dataset = random_split(
            dataset,
            [train_size, validate_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader, val_loader, test_loader = (
            DataLoader(
                train_dataset,
                BATCH_SIZE,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            ),
            DataLoader(
                validate_dataset,
                BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            ),
            DataLoader(
                test_dataset,
                BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            ),
        )

        model = ClassificationNetwork().to(DEVICE)

        early_stopper = EarlyStopping(
            patience=15, delta=0.001, save_path="best_model.pt"
        )
        logging.info(
            'early stopper: EarlyStopping(patience=15, delta=0.001, save_path="best_model.pt")'
        )

        logging.info(f"model: {model}")
        loss_fn = nn.CrossEntropyLoss()
        logging.info(f"loss functin: {loss_fn}")
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        logging.info(f"optimizer: {optimizer}")

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.0001,
        )
        logging.info(
            'scheduler: ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=5,threshold=0.0001,verbose=True,)'
        )

        TAG_MAPPING: dict[str, int] = {
            tag: index for index, tag in enumerate(embedding_model.dv.index_to_key)
        }  # type: ignore

        train_losses = []
        train_first_match_accs = []
        train_second_match_accs = []

        val_losses = []
        val_first_match_accs = []
        val_second_match_accs = []
        logging.info("Model Training ...")
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_epoch_loss, train_first_match_acc, train_second_match_acc = (
                Classifier.train(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    scheduler,
                    TAG_MAPPING,
                    epoch,
                )
            )

            train_losses.append(train_epoch_loss)
            train_first_match_accs.append(train_first_match_acc)
            train_second_match_accs.append(train_second_match_acc)
            logging.info(
                f"Train - Batch Size {BATCH_SIZE}, RL {LEARNING_RATE if not scheduler else scheduler.get_last_lr()}, Epoch [{epoch + 1}/{EPOCHS}] Loss {train_epoch_loss:>10.5f}"
            )

            # Validation
            model.eval()
            val_epoch_loss, val_first_match_acc, val_second_match_acc = (
                Classifier.validate(model, val_loader, loss_fn, TAG_MAPPING)
            )
            logging.info(
                f"Validation - Batch Size {BATCH_SIZE}, RL {LEARNING_RATE if not scheduler else scheduler.get_last_lr()}, Epoch [{epoch + 1}/{EPOCHS}] Loss {val_epoch_loss:>10.5f}"
            )
            val_losses.append(val_epoch_loss)
            val_first_match_accs.append(val_first_match_acc)
            val_second_match_accs.append(val_second_match_acc)

            early_stopper(val_epoch_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

            if scheduler is not None:
                scheduler.step(val_epoch_loss)

        Classifier.plot_learning_curves(
            train_losses,
            val_losses,
            train_first_match_accs,
            val_first_match_accs,
            train_second_match_accs,
            val_second_match_accs,
        )

        model.load_state_dict(torch.load("best_model.pt"))  # 如果有 early stop
        # Evaluate after training
        model.eval()
        test_first_accuracy, test_second_accuracy = Classifier.evaluate(
            model, test_loader, TAG_MAPPING
        )

        torch.save(
            model,
            f"{BASE_PATH}{EXECUTE_AT}-first-{int(test_first_accuracy * 100)}-second-{int(test_second_accuracy * 100)}.classify.model",
        )

    @staticmethod
    def train(
        model, train_loader, loss_fn, optimizer, scheduler, TAG_MAPPING, current_epoch
    ):
        loop = tqdm(train_loader, total=len(train_loader))
        train_first_match = 0
        train_second_match = 0
        train_epoch_loss = 0
        train_total_samples = 0

        for labels, features in loop:
            features = features.to(DEVICE)
            targets = Classifier.one_hot_encoding(labels, TAG_MAPPING, DEVICE)

            optimizer.zero_grad()

            prediction = model(features)
            loss = loss_fn(prediction, targets)
            loss.backward()
            optimizer.step()

            first_match, second_match = Classifier.calculate_top_k_accuracy(
                prediction, labels, TAG_MAPPING, DEVICE
            )

            loop.set_description(
                f"Batch Size {BATCH_SIZE}, RL {LEARNING_RATE if not scheduler else scheduler.get_last_lr()}, Epoch [{current_epoch + 1}/{EPOCHS}] Loss {loss.item():>10.5}"
            )

            train_first_match += first_match
            train_second_match += second_match
            train_total_samples += len(labels)
            train_epoch_loss += loss.item() * len(labels)

        train_epoch_loss /= train_total_samples
        train_first_match_acc = train_first_match / train_total_samples
        train_second_match_acc = train_second_match / train_total_samples
        return train_epoch_loss, train_first_match_acc, train_second_match_acc

    @staticmethod
    @torch.no_grad()
    def validate(model, val_loader, loss_fn, TAG_MAPPING):
        val_first_match = 0
        val_second_match = 0
        val_epoch_loss = 0
        val_total_samples = 0
        for labels, features in val_loader:
            features = features.to(DEVICE)
            targets = Classifier.one_hot_encoding(labels, TAG_MAPPING, DEVICE)

            prediction = model(features)
            loss = loss_fn(prediction, targets)

            first_match, second_match = Classifier.calculate_top_k_accuracy(
                prediction, labels, TAG_MAPPING, DEVICE
            )

            val_first_match += first_match
            val_second_match += second_match
            val_total_samples += len(labels)
            val_epoch_loss += loss.item() * len(labels)

        val_epoch_loss /= val_total_samples
        val_first_acc = val_first_match / val_total_samples
        val_second_acc = val_second_match / val_total_samples

        return val_epoch_loss, val_first_acc, val_second_acc

    @staticmethod
    @torch.no_grad()
    def evaluate(
        model,
        test_loader,
        TAG_MAPPING,
        topk_pred_labels_dict=None,
        target_labels=None,
    ):
        test_first_match = 0
        test_second_match = 0
        test_total_samples = 0
        logging.info("Model Evaluating After Training ...")

        for labels, features in test_loader:
            features = features.to(DEVICE)
            prediction = model(features)

            first_match, second_match = Classifier.calculate_top_k_accuracy(
                prediction, labels, TAG_MAPPING, DEVICE
            )

            if topk_pred_labels_dict is not None and target_labels is not None:
                # for confusion matrix
                topk_pred_labels_dict[1].extend(prediction.topk(1).indices)
                topk_pred_labels_dict[2].extend(prediction.topk(2).indices)
                target_labels.extend(labels)

            test_first_match += first_match
            test_second_match += second_match
            test_total_samples += len(labels)

        test_first_accuracy = test_first_match / test_total_samples
        test_second_accuracy = test_second_match / test_total_samples
        logging.info(f"1st-Match Accuracy after traning: {test_first_accuracy}")
        logging.info(f"2nd-Match Accuracy after traning: {test_second_accuracy}")
        return test_first_accuracy, test_second_accuracy

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
    def get_validation_epochs(
        epochs: int, num_validation_points: int = 100
    ) -> list[int]:
        if epochs <= num_validation_points:
            validation_epochs = list(range(epochs))
        else:
            step = (
                epochs // (num_validation_points - 1)
            )  # (num_validation_points - 1) points + the final epoch = num_validation_points total validation points
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
        validation_epochs=None,
        save_path=LEARNING_CURVE_PATH,
    ):
        plt.figure(figsize=(18, 5))

        x_values = (
            validation_epochs
            if validation_epochs is not None
            else range(len(train_losses))
        )

        plt.subplot(1, 3, 1)
        plt.plot(x_values, train_losses, label="Training Loss")
        plt.plot(x_values, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(x_values, train_first_match_accuracies, label="Training Accuracy")
        plt.plot(x_values, val_first_match_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("1st-Matched Accuracy Curves")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(x_values, train_second_match_accuracies, label="Training Accuracy")
        plt.plot(x_values, val_second_match_accuracies, label="Validation Accuracy")
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
    Classifier.run()
