from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim

DEFAULT_FILE_PATH = "week-7/dataset/gender-height-weight.csv"


class GenderHeightWeightDataset(Dataset):
    def __init__(
        self,
        file_path=DEFAULT_FILE_PATH,
    ):
        self.raw_data: pd.DataFrame = pd.read_csv(file_path)

        self.height_mean = self.raw_data["Height"].mean()
        self.height_std = self.raw_data["Height"].std(ddof=0)
        self.weight_mean = self.raw_data["Weight"].mean()
        self.weight_std = self.raw_data["Weight"].std(ddof=0)

        self.features = np.column_stack(
            (
                np.where(self.raw_data["Gender"] == "Male", 1.0, 0.0).astype(
                    np.float32
                ),
                (
                    (self.raw_data["Height"].to_numpy() - self.height_mean)
                    / self.height_std
                ).astype(np.float32),
            )
        )

        self.labels = (
            (self.raw_data["Weight"].to_numpy() - self.weight_mean) / self.weight_std
        ).astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return features, label


class GenderHeightWeightNetwork(nn.Module):
    def __init__(self):
        super(GenderHeightWeightNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.model(x)


class Task1:
    @staticmethod
    def run():
        print(f"{' Task 1: Weight Prediction ':=^75}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {str.upper(device.type)}")
        dataset = GenderHeightWeightDataset()

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        base_lr = 0.001  # lr when batch_size = 1
        batch_size = 1
        learning_rate = base_lr * (batch_size**0.5)  # based on SGD

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size)

        model = GenderHeightWeightNetwork().to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Evaluate before training
        model.eval()
        total_loss_in_weights = 0.0
        with torch.no_grad():
            for _, (features, labels) in enumerate(test_loader):
                features = features.to(device)
                labels = labels.to(device)

                output = model(features)
                loss = loss_fn(output, labels)
                total_loss_in_weights += loss.item() ** 0.5 * dataset.weight_std

        avg_loss_in_weights = total_loss_in_weights / len(test_loader)
        print(f"Before Training Avg Loss in Weight: {avg_loss_in_weights: .2f} pounds")

        # Training
        model.train()
        epochs = 15
        print("Model Training ...")
        for _ in range(epochs):
            total_loss_in_weights = 0.0
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                prediction = model(features)
                loss = loss_fn(prediction, labels)
                loss.backward()
                optimizer.step()

                total_loss_in_weights += loss.item() ** 0.5 * dataset.weight_std

            avg_loss_in_weights = total_loss_in_weights / len(train_loader)
        total_loss_in_weights = 0.0

        # Evaluate after training
        model.eval()
        with torch.no_grad():
            for _, (features, labels) in enumerate(test_loader):
                features = features.to(device)
                labels = labels.to(device)
                output = model(features)
                loss = loss_fn(output, labels)
                total_loss_in_weights += loss.item() ** 0.5 * dataset.weight_std
        avg_loss_in_weights = total_loss_in_weights / len(test_loader)
        print(f"After Training Avg Loss in Weight: {avg_loss_in_weights: .2f} pounds")
        print()
