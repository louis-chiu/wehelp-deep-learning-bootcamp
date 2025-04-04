import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from corpus_utils import spllit_data
from embedding import setup_model_configuration
from datetime import datetime
import logging

BASE_PATH = "./" # "./0327-1503/"
PATH = f"{BASE_PATH}example-data.csv"
EMBEDDING_MODEL_PATH = f"{BASE_PATH}0402-0002-75-86.model"
EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Task:
    @staticmethod
    def run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Training Using {device}")
        base_lr = 0.005
        batch_size = 8
        learning_rate = base_lr * (batch_size**0.5)
        embedding_model = setup_model_configuration(path=EMBEDDING_MODEL_PATH)

        train_dataset, test_dataset = spllit_data(PATH)
        train_dataset = [
            (label, torch.from_numpy(embedding_model.infer_vector(feature)))
            for label, *feature in train_dataset
        ]
        test_dataset = [
            (label, torch.from_numpy(embedding_model.infer_vector(feature)))
            for label, *feature in test_dataset
        ]

        model = ClassificationNetwork().to(device)
        loss_fn = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        TAG_MAPPING = {
            tag: index for index, tag in enumerate(embedding_model.dv.index_to_key)
        }

        # Evaluate before training
        model.eval()
        correct_times = 0
        logging.info("Model Evaluating Before Training ...")
        with torch.no_grad():
            for ephc, (label, feature) in enumerate(test_dataset):
                feature = feature.to(device)
                target = torch.as_tensor(
                    Task.one_hot_encoding(label, TAG_MAPPING.keys()),
                    dtype=torch.float,
                ).to(device)

                output = model(feature)
                loss = loss_fn(output, target)
                if Task.is_prediction_correct(output, TAG_MAPPING.get(label)):
                    correct_times += 1

        accuracy = correct_times / len(test_dataset)
        logging.info(f"Accuracy before traning: {accuracy}")

        # Training
        model.train()
        epochs = 100
        logging.info("Model Training ...")
        for epoch in range(epochs):
            loop = tqdm(enumerate(train_dataset), total=len(train_dataset))

            for i, (label, feature) in loop:
                feature = feature.to(device)
                target = torch.as_tensor(
                    Task.one_hot_encoding(label, TAG_MAPPING.keys()),
                    dtype=torch.float,
                ).to(device)

                optimizer.zero_grad()

                prediction = model(feature)
                loss = loss_fn(prediction, target)
                loss.backward()
                optimizer.step()

                loop.set_description(
                    f"Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():^10.5}"
                )
                
                if i == len(train_dataset) - 1:
                    logging.info(
                        f"Batch Size {batch_size}, RL {learning_rate}, Epoch [{epoch + 1}/{epochs}] Loss {loss.item():^10.5}"
                    )

        # Evaluate after training
        model.eval()
        correct_times = 0
        logging.info("Model Evaluating After Training ...")
        with torch.no_grad():
            for _, (label, feature) in enumerate(test_dataset):
                feature = feature.to(device)
                target = torch.as_tensor(
                    Task.one_hot_encoding(label, TAG_MAPPING.keys()),
                    dtype=torch.float,
                ).to(device)

                output = model(feature)
                loss = loss_fn(output, target)
                if Task.is_prediction_correct(output, TAG_MAPPING.get(label)):
                    correct_times += 1
        accuracy = correct_times / len(test_dataset)
        logging.info(f"Accuracy after traning: {accuracy}")

        torch.save(
            model.state_dict(), f"{BASE_PATH}{EXECUTE_AT}-{int(accuracy * 100)}.classify.model"
        )

    @staticmethod
    def is_prediction_correct(output, target_index) -> bool:
        return output.argmax().item() == target_index

    @staticmethod
    def one_hot_encoding(target, labels):
        return [0 if target != label else 1 for label in labels]


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=f"{BASE_PATH}classify-{EXECUTE_AT}.log",
    )
    Task.run()
