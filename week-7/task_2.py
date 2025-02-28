from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim

DEFAULT_FILE_PATH = "week-7/dataset/titanic.csv"


class TitanicDataset(Dataset):
    def __init__(
        self,
        file_path=DEFAULT_FILE_PATH,
    ):
        self.raw_data: pd.DataFrame = pd.read_csv(file_path)

        (
            _,
            IS_SURVIVED,
            PCLASS,
            _,
            SEX,
            AGE,
            SIB_SP,
            PARCH,
            _,
            FARE,
            _,
            _,
        ) = self.raw_data.columns
        FAMILY_NUMS = "family_nums"

        self.raw_data[FAMILY_NUMS] = self.raw_data[SIB_SP].fillna(0) + self.raw_data[
            FARE
        ].fillna(0)

        self.age_mean = self.raw_data[AGE].mean()
        self.age_std = self.raw_data[AGE].std(ddof=0)
        self.family_nums_mean = self.raw_data[FAMILY_NUMS].mean()
        self.family_nums_std = self.raw_data[FAMILY_NUMS].std(ddof=0)
        self.fare_mean = self.raw_data[FARE].mean()
        self.fare_std = self.raw_data[FARE].std(ddof=0)

        self.features = np.column_stack(
            [
                np.where(self.raw_data[PCLASS] == 1, 1, 0).astype(np.float32),
                np.where(self.raw_data[PCLASS] == 2, 1, 0).astype(np.float32),
                np.where(self.raw_data[PCLASS] == 3, 1, 0).astype(np.float32),
                self.raw_data[SEX].map({"male": 1, "female": 0}).fillna(-1),
                (
                    (self.raw_data[AGE].fillna(self.age_mean) - self.age_mean)
                    / self.age_std
                ).to_numpy(dtype=np.float32),
                (
                    (self.raw_data[FAMILY_NUMS] - self.family_nums_mean)
                    / self.family_nums_std
                ).to_numpy(dtype=np.float32),
                ((self.raw_data[FARE] - self.fare_mean) / self.fare_std).to_numpy(
                    dtype=np.float32
                ),
            ]
        )

        self.labels = self.raw_data[IS_SURVIVED].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return features, label


class TitanicNetwork(nn.Module):
    def __init__(self):
        super(TitanicNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Task2:
    @staticmethod
    def run():
        print(f"{' Task 2: Weight Prediction ':=^75}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {str.upper(device.type)}")
        dataset = TitanicDataset()

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        base_lr = 0.005  # lr when batch_size = 1
        batch_size = 1
        learning_rate = base_lr * (batch_size**0.5)  # based on SGD

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size)

        model = TitanicNetwork().to(device)
        loss_fn = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Evaluate before training
        model.eval()
        correct_times = 0
        with torch.no_grad():
            for _, (features, labels) in enumerate(test_loader):
                features = features.to(device)
                labels = labels.to(device)

                output = model(features)

                is_survived = 1 if output[0] > 0.5 else 0
                if is_survived == labels[0]:
                    correct_times += 1

        accuracy = correct_times / len(test_loader)
        print(f"Before Training Prediction Accuracy: {accuracy * 100: .2f} %")

        # Training
        model.train()
        correct_times = 0
        epochs = 20
        print("Model Training ...")
        for _ in range(epochs):
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                prediction = model(features)
                loss = loss_fn(prediction, labels)
                loss.backward()
                optimizer.step()

        # Evaluate after training
        model.eval()
        correct_times = 0
        with torch.no_grad():
            for _, (features, labels) in enumerate(test_loader):
                features = features.to(device)
                labels = labels.to(device)

                output = model(features)

                is_survived = 1 if output[0] > 0.5 else 0
                if is_survived == labels[0]:
                    correct_times += 1

        accuracy = correct_times / len(test_loader)
        print(f"After Training Prediction Accuracy: {accuracy * 100: .2f} %")
        print()
