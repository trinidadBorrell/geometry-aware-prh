"""exp-016: D2 = the exp-006 WIT-1M subset with each image's Gemini recaption added.

exp-006 took the first 10240 rows of `askoepke/wit_1m_recaptioned` (config `wit_1m`) with a
non-empty original caption and stored image bytes, caption and url. This streams the same rows
again (text columns only), checks that the urls agree row by row, and adds `gemini_caption`.

    uv run python experiments/exp-016-positive-controls/prepare_d2.py \\
        --src /workspace/results/oddharak/exp-006-sample-size/wit_1m_first10240.parquet \\
        --out /workspace/results/oddharak/exp-016-positive-controls/d2.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset

    df = pd.read_parquet(args.src)
    stream = load_dataset("askoepke/wit_1m_recaptioned", "wit_1m", split="train", streaming=True)
    stream = stream.select_columns(["original_caption", "gemini_caption", "url"])
    gem = []
    for row in stream:
        if not (row["original_caption"] or "").strip():
            continue
        gem.append((row["url"], (row["gemini_caption"] or "").strip()))
        if len(gem) == len(df):
            break
    urls = [u for u, _ in gem]
    assert urls == df["url"].tolist(), "stream order differs from the exp-006 parquet"
    df["gemini_caption"] = [g for _, g in gem]
    empty = int((df["gemini_caption"] == "").sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(
        f"saved {len(df)} rows to {args.out}; empty Gemini captions: {empty}; "
        f"median length original {df['caption'].str.len().median():.0f}, "
        f"Gemini {df['gemini_caption'].str.len().median():.0f} characters"
    )


if __name__ == "__main__":
    main()
