import argparse
import os
import random
import torch
import pandas as pd
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Load the model's state_dict
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Draw bounding boxes on an image using matplotlib
def draw_bounding_boxes(image_path, boxes, labels, output_path, show_label):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        # Draw the bounding box
        rect = patches.Rectangle(
            (box[0], box[1]),  # (x, y)
            box[2] - box[0],  # width
            box[3] - box[1],  # height
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add the label if required
        if show_label:
            ax.text(
                box[0],
                box[1] - 5,  # Slightly above the box
                label[:4],  # Limit to the first 4 characters
                color="#FF00FF",
                fontsize=8,
            )

    # Save the output image
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on an image using a trained model."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model's state_dict file."
    )
    parser.add_argument(
        "--labels", type=str, required=True, help="Path to the test_labels.csv file."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific image file to process. If not provided, a random image will be selected.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Directory to save the output image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for filtering bounding boxes (0 to 1). If not set, no filtering is applied.",
    )
    parser.add_argument(
        "--show-label",
        action="store_true",
        help="Flag to determine whether to display labels on the image. If set, labels will be displayed.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to process. Default is 1.",
    )
    args = parser.parse_args()

    # Load the model
    num_classes = 6  # Adjust this based on your dataset
    model = load_model(args.model, num_classes)

    # Load the labels
    labels_df = pd.read_csv(args.labels)

    # Select images
    selected_images = []
    if args.image:
        selected_images = [args.image]
    else:
        unique_filenames = list(labels_df["filename"].unique())
        if len(unique_filenames) < args.count:
            raise ValueError(
                "The number of unique images is less than the requested count."
            )
        selected_images = random.sample(unique_filenames, args.count)

    for selected_image in selected_images:
        # Get the image path
        image_path = os.path.join(os.path.dirname(args.labels), "test", selected_image)

        # Prepare the image for the model
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(torch.float32)

        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)

        # Extract bounding boxes, labels, and apply threshold if specified
        boxes = outputs[0]["boxes"].cpu().numpy().tolist()
        scores = outputs[0]["scores"].cpu().numpy().tolist()
        labels = outputs[0]["labels"].cpu().numpy().tolist()

        if args.threshold is not None:
            filtered_boxes = []
            filtered_labels = []
            for box, score, label in zip(boxes, scores, labels):
                if score >= args.threshold:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
            boxes = filtered_boxes
            labels = filtered_labels

        # Map label indices to class names
        class_names = {1: "Bus", 2: "Car", 3: "Motor", 4: "Pickup", 5: "Truck"}
        label_names = [class_names[label] for label in labels]

        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)

        # Save the image with bounding boxes and labels
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_filename = f"{timestamp}_{os.path.basename(selected_image)}"
        output_path = os.path.join(args.output, output_filename)
        draw_bounding_boxes(
            image_path, boxes, label_names, output_path, args.show_label
        )
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
