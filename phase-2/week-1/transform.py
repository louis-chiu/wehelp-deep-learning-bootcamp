import pandas as pd
from pathlib import Path

SOURCE_DIR = Path("./data")
OUTPUT_DIR = Path("./clean_data")
BOARD_NAMES = [
    "baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]


def transform(board_name):
    df = pd.read_csv(SOURCE_DIR / f"{board_name}.csv")

    df = df[~df["title"].str.startswith(("Re:", "Fw:"))]

    df = df.apply(lambda x: x.str.strip()).apply(lambda x: x.str.lower())

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_DIR / f"{board_name}.csv", index=False)


def main():
    for board_name in BOARD_NAMES:
        transform(board_name)


if __name__ == "__main__":
    main()
