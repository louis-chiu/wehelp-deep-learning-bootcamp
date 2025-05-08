import logging
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

EXECUTE_AT = datetime.now()
OUTPUT_PATH = f"logs/{EXECUTE_AT}.log"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename=OUTPUT_PATH,
)


class HandWritingDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.dataset = ImageFolder(path, transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        return self.dataset[index]


class Network(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()

        self.features = nn.Sequential(
            # 64 * 64 * 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32 * 32 * 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 16 * 16 * 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 8 * 8 * 128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4 * 4 * 256
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, num_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():
    BASE_PATH = ""  # "phase-3/week-1/"
    TRAIN_PATH = f"{BASE_PATH}data/handwriting/augmented_images/augmented_images1"
    TEST_PATH = f"{BASE_PATH}data/handwriting/handwritten-english-characters-and-digits/combined_folder/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {str.upper(device.type)}")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_dataset, test_dataset = (
        HandWritingDataset(TRAIN_PATH, transform),
        HandWritingDataset(TEST_PATH, transform),
    )

    batch_size = 32
    learning_rate = 0.001

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=10, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size, num_workers=10, pin_memory=True)

    model = Network(num_of_classes=len(train_dataset.classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    logging.info(f"model: {model}")
    loss_fn = nn.CrossEntropyLoss()

    # Training
    model.train()
    correct_times = 0
    epochs = 20
    logging.info("Model Training ...")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for i, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            prediction = model(features)

            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)

            running_loss = (
                0.9 * running_loss + 0.1 * loss.item() if i > 0 else loss.item()
            )

            pbar.set_postfix({"loss": f"{running_loss:.4f}"})

    # Evaluate after training
    model.eval()
    correct_times = 0
    total_samples = 0
    logging.info("Model Evaluating ...")
    with torch.no_grad():
        for _, (features, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            features = features.to(device)
            labels = labels.to(device)

            prediction = model(features)
            _, predicted = torch.max(prediction, dim=1)
            correct_times += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_times / total_samples
    logging.info(f"After Training Prediction Accuracy: {accuracy * 100: .2f} %")


if __name__ == "__main__":
    main()
