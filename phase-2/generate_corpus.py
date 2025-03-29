from corpus_utils import read_as_tagged_documents
import argparse


PATH = "example-data.csv"
OUTPUT_PATH = "corpus.cor"


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
    output_path = args.output_path if args.output_path else PATH

    # 將 TaggedDocument 寫入 corpus 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in read_as_tagged_documents(path):
            line = " ".join(doc.words) + "\n"
            f.write(line)


if __name__ == "__main__":
    main()
