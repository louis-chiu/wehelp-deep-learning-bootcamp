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

    return df

def main():
    combined_df = pd.DataFrame()
    for board_name in BOARD_NAMES:
        df = transform(board_name)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    print(f"Total data size: {combined_df.shape[0]}")
    combined_df.to_csv(OUTPUT_DIR / "combined.csv", index=False)


if __name__ == "__main__":
    main()
