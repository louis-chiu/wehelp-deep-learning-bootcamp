import argparse
import torch
from torch import nn

BASE_PATH = "./0327-1503/"
PATH = f"{BASE_PATH}0414-1830-first-88-second-95.classify.model"
OUTPUT_PATH = f"{PATH}.log"


# for laoding model
class ClassificationNetwork(nn.Module): ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the tokenized document file for generating the corpus file.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="path to save the corpus file",
        type=str,
    )
    args = parser.parse_args()

    path = args.path if args.path else PATH
    output_path = args.output_path if args.output_path else OUTPUT_PATH

    model = torch.load(path, weights_only=False)
    
    torch.save(model.state_dict(), "0414-1830-88-95.classify.model")
if __name__ == "__main__":
    main()
