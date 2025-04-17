from classify import ClassificationNetwork, Classifier, PTTDataset
from utils import ModelUtils

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob
import numpy as np

BASE_PATH = "./0327-1503/"
MODEL_NAME = "0414-0810-first-88-second-95"
PATH = f"{BASE_PATH}{MODEL_NAME}.classify.model"  # f"{BASE_PATH}example.pt"
EMBEDDING_MODEL_PATH = f"{BASE_PATH}0402-0002-75-86.model"
SAVE_PATH = f"{BASE_PATH}{MODEL_NAME}-confusion-matrix.png"
BATCH_SIZE = 128


def plot_confusion_matrix(
    y_true, y_pred, y_pred_top2, class_names, save_path=SAVE_PATH
):
    # Calculate confusion matrices for top-1 predictions
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent_total = cm.astype("float") / cm.sum() * 100

    # Find indices where top-1 prediction was wrong
    incorrect_indices = [
        i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred
    ]

    # Get true labels and top-2 predictions for those incorrect samples
    incorrect_true = [y_true[i] for i in incorrect_indices]
    incorrect_pred_top2 = [y_pred_top2[i] for i in incorrect_indices]

    # Calculate confusion matrix for top-2 predictions on incorrect samples
    cm_top2_incorrect = confusion_matrix(incorrect_true, incorrect_pred_top2)

    # Calculate percentage of true labels for top-2 incorrect predictions
    cm_top2_percent = (
        cm_top2_incorrect.astype("float")
        / cm_top2_incorrect.sum(axis=1)[:, np.newaxis]
        * 100
    )
    cm_top2_percent = np.nan_to_num(cm_top2_percent)  # Replace NaN with 0

    # Calculate percentage of total incorrect dataset for top-2
    total_incorrect = cm_top2_incorrect.sum()
    cm_top2_percent_total = (
        cm_top2_incorrect.astype("float") / total_incorrect * 100
        if total_incorrect > 0
        else np.zeros_like(cm_top2_incorrect)
    )

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(48, 24))

    # Plot 1: Actual counts (top-1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 0],
    )
    axes[0, 0].set_xlabel("Predicted Labels")
    axes[0, 0].set_ylabel("True Labels")
    axes[0, 0].set_title("Confusion Matrix - Top-1 (Counts)")

    # Plot 2: Percentages of true labels (top-1)
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("Predicted Labels")
    axes[0, 1].set_ylabel("True Labels")
    axes[0, 1].set_title("Confusion Matrix - Top-1 (% of True Labels)")

    # Plot 3: Percentages of total dataset (top-1)
    sns.heatmap(
        cm_percent_total,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 2],
    )
    axes[0, 2].set_xlabel("Predicted Labels")
    axes[0, 2].set_ylabel("True Labels")
    axes[0, 2].set_title("Confusion Matrix - Top-1 (% of Total Dataset)")

    # Plot 4: Actual counts (top-2 for incorrect predictions)
    sns.heatmap(
        cm_top2_incorrect,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("Top-2 Predicted Labels")
    axes[1, 0].set_ylabel("True Labels")
    axes[1, 0].set_title("Confusion Matrix - Top-2 (Counts for Top-1 Incorrect)")

    # Plot 5: Percentages of true labels (top-2 for incorrect predictions)
    sns.heatmap(
        cm_top2_percent,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("Top-2 Predicted Labels")
    axes[1, 1].set_ylabel("True Labels")
    axes[1, 1].set_title(
        "Confusion Matrix - Top-2 (% of True Labels for Top-1 Incorrect)"
    )

    # Plot 6: Percentages of total incorrect dataset (top-2)
    sns.heatmap(
        cm_top2_percent_total,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 2],
    )
    axes[1, 2].set_xlabel("Top-2 Predicted Labels")
    axes[1, 2].set_ylabel("True Labels")
    axes[1, 2].set_title("Confusion Matrix - Top-2 (% of Top-1 Incorrect Predictions)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    return cm


def plot_confusion_matrix_four_pic(
    y_true, y_pred, y_pred_top2, class_names, save_path=SAVE_PATH
):
    # Calculate confusion matrices for top-1 predictions
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent_total = cm.astype("float") / cm.sum() * 100

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    # Plot 1: Actual counts (top-1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 0],
    )
    axes[0, 0].set_xlabel("Predicted Labels")
    axes[0, 0].set_ylabel("True Labels")
    axes[0, 0].set_title("Confusion Matrix - Top-1 (Counts)")

    # Plot 2: Percentages of true labels (top-1)
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("Predicted Labels")
    axes[0, 1].set_ylabel("True Labels")
    axes[0, 1].set_title("Confusion Matrix - Top-1 (% of True Labels)")

    # Plot 3: Percentages of total dataset (top-1)
    sns.heatmap(
        cm_percent_total,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("Predicted Labels")
    axes[1, 0].set_ylabel("True Labels")
    axes[1, 0].set_title("Confusion Matrix - Top-1 (% of Total Dataset)")

    # Plot 4: Top-2 predictions for samples where top-1 was incorrect
    # Find indices where top-1 prediction was wrong
    incorrect_indices = [
        i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred
    ]

    # Get true labels and top-2 predictions for those incorrect samples
    incorrect_true = [y_true[i] for i in incorrect_indices]
    incorrect_pred_top2 = [y_pred_top2[i] for i in incorrect_indices]

    # Calculate confusion matrix for top-2 predictions on incorrect samples
    cm_top2_incorrect = confusion_matrix(incorrect_true, incorrect_pred_top2)

    sns.heatmap(
        cm_top2_incorrect,
        annot=True,
        fmt="d",
        cmap="Greens",  # Different colormap to distinguish
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("Top-2 Predicted Labels")
    axes[1, 1].set_ylabel("True Labels")
    axes[1, 1].set_title("Confusion Matrix - Top-2 (For Top-1 Incorrect Predictions)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    return cm


def plot_confusion_matrix_three_pic(y_true, y_pred, class_names, save_path=SAVE_PATH):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentage of true labels
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Calculate percentage of total dataset
    cm_percent_total = cm.astype("float") / cm.sum() * 100

    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 8))

    # Plot actual numbers on the left
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )
    ax1.set_xlabel("Predicted Labels")
    ax1.set_ylabel("True Labels")
    ax1.set_title("Confusion Matrix (Counts)")

    # Plot percentages of true labels in the middle
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",  # Display to one decimal place
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
    )
    ax2.set_xlabel("Predicted Labels")
    ax2.set_ylabel("True Labels")
    ax2.set_title("Confusion Matrix (% of True Labels)")

    # Plot percentages of total dataset on the right
    sns.heatmap(
        cm_percent_total,
        annot=True,
        fmt=".1f",  # Display to one decimal place
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax3,
    )
    ax3.set_xlabel("Predicted Labels")
    ax3.set_ylabel("True Labels")
    ax3.set_title("Confusion Matrix (% of Total Dataset)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    return cm


def get_test_data_set(embedding_model):
    dataset = PTTDataset(embedding_model)
    train_size = int(0.8 * len(dataset))
    validate_size = int((len(dataset) - train_size) / 2)
    test_size = len(dataset) - train_size - validate_size

    _, _, test_dataset = random_split(
        dataset,
        [train_size, validate_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return test_dataset


def to_test_loader(test_dataset):
    return DataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )


def main(model_path, embedding_model, test_loader):
    model = torch.load(model_path, weights_only=False)

    TAG_MAPPING: dict[str, int] = {
        tag: index for index, tag in enumerate(embedding_model.dv.index_to_key)
    }  # type: ignore

    TAG_MAPPING_REV: dict[str, int] = {
        index: tag for index, tag in enumerate(embedding_model.dv.index_to_key)
    }  # type: ignore

    topk_pred_labels_dict = {1: [], 2: []}
    target_labels = []

    Classifier.evaluate(
        model, test_loader, TAG_MAPPING, topk_pred_labels_dict, target_labels
    )

    plot_confusion_matrix(
        target_labels,
        [TAG_MAPPING_REV.get(index.item()) for index in topk_pred_labels_dict[1]],
        [
            TAG_MAPPING_REV.get(index2.item())
            for index1, index2 in topk_pred_labels_dict[2]
        ],
        embedding_model.dv.index_to_key,
        save_path=f"{model_path}-confusion-matrix.png",
    )


if __name__ == "__main__":
    # Execute a command equivalent to "ls -l 0327-1503/*first*second*.classify.model"

    model_path
    # model_paths = glob.glob("0327-1503/*first*second*.classify.model")
    embedding_model = ModelUtils.setup_model_configuration(EMBEDDING_MODEL_PATH)
    test_dataset = get_test_data_set(embedding_model)

    for model_path in model_paths:
        test_loader = to_test_loader(test_dataset)
        print(model_path)
        main(model_path, embedding_model, test_loader)
