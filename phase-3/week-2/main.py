import logging
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

BASE_PATH = ""  # "phase-3/week-2/"
IMAGE_BASE_PATH = f"{BASE_PATH}data/r-cnn-data/"
EXECUTE_AT = datetime.now()
OUTPUT_PATH = f"{BASE_PATH}logs/{EXECUTE_AT}.log"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename=OUTPUT_PATH,
)


class VehicleDataset(Dataset):
    def __init__(self, mode="train", use_preprocessed=False):
        super().__init__()
        self.mode = mode
        self.use_preprocessed = use_preprocessed
        raw_labels = pd.read_csv(
            f"{BASE_PATH}data/r-cnn-data/vehicles_images/{mode}_labels.csv"
        )
        grouped = (
            raw_labels.groupby("filename", group_keys=False)
            .apply(
                lambda x: {
                    "boxes": [
                        [
                            row["xmin"],
                            row["ymin"],
                            row["xmax"],
                            row["ymax"],
                        ]
                        for _, row in x.iterrows()
                    ],
                    "labels": [row["class"] for _, row in x.iterrows()],
                }
            )
            .reset_index()
        )
        self.dataset = grouped.rename(columns={0: "labels"})
        self.class_to_idx = {
            "Bus": 1,
            "Car": 2,
            "Motorcycle": 3,
            "Pickup": 4,
            "Truck": 5,
        }
        self.classes = (self.class_to_idx).keys()

        self.transform = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        filename = self.dataset.loc[index, "filename"]
        img_path = f"{IMAGE_BASE_PATH}vehicles_images/{self.mode}/{filename}"
        img = self.transform(Image.open(img_path).convert("RGB"))

        label_info = self.dataset.loc[index, "labels"]
        boxes = torch.tensor(label_info["boxes"], dtype=torch.float32)
        labels = torch.tensor(
            [self.class_to_idx[label] for label in label_info["labels"]],
            dtype=torch.int64,
        )
        target = {"boxes": boxes, "labels": labels}
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def main():
    logging.info(f"Using {str.upper(device.type)}")

    train_dataset, test_dataset = (
        VehicleDataset(mode="train"),
        VehicleDataset(mode="test"),
    )

    batch_size = 2
    learning_rate = 0.0001

    train_loader, test_loader = (
        DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            collate_fn=collate_fn,
        ),
        DataLoader(
            test_dataset,
            batch_size,
            num_workers=10,
            pin_memory=True,
            collate_fn=collate_fn,
        ),
    )

    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    num_classes = 6

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    model = model.to(device)
    logging.info(f"model: {model}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for name, param in model.backbone.body.named_parameters():
        if "layer1" in name or "layer2" in name:  # or "layer3" in name:
            param.requires_grad = False

    # Training
    model.train()
    epochs = 250
    logging.info("Model Training ...")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for i, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            running_loss = (
                0.9 * running_loss + 0.1 * loss.item() if i > 0 else loss.item()
            )
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})
        logging.info(f"Epoch {epoch}/{epochs}, Loss: {running_loss}")

    save_model(model, f"{BASE_PATH}outputs/{EXECUTE_AT}.pt")

    # Evaluating
    model.eval()
    logging.info("Model Evaluating ...")
    total_score = 0.0
    total_objects = 0
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            outputs = model(images)
            for output in outputs:
                scores = output["scores"].cpu().numpy()
                total_score += scores.sum()
                total_objects += len(scores)
        avg_score = total_score / total_objects if total_objects > 0 else 0.0
        logging.info(f"Average score: {avg_score * 100:.2f}")


if __name__ == "__main__":
    main()
