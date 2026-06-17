#!/usr/bin/env python3
###python visualize_r2.py --log_dir /path/to/logs --output r2_vs_epoch.png

import json
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def get_epoch_from_filename(path: Path) -> int:
    """
    Extract epoch number from filenames like:
    train_log_epoch37.json
    """
    match = re.search(r"epoch(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not extract epoch from filename: {path.name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="Directory containing train_log_epoch*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="r2_vs_epoch.png",
        help="Output figure name",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    files = sorted(
        log_dir.glob("train_log_epoch*.json"),
        key=get_epoch_from_filename,
    )

    if len(files) == 0:
        raise FileNotFoundError(f"No train_log_epoch*.json files found in {log_dir}")

    epochs = []
    train_r2 = []
    valid_r2 = []

    for file in files:
        epoch = get_epoch_from_filename(file)

        with open(file, "r") as f:
            log = json.load(f)

        if "train_r2" not in log or "valid_r2" not in log:
            print(f"Skipping {file.name}: missing train_r2 or valid_r2")
            continue

        epochs.append(epoch)
        train_r2.append(log["train_r2"])
        valid_r2.append(log["valid_r2"])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_r2, marker="o", label="Train R2")
    plt.plot(epochs, valid_r2, marker="s", label="Valid R2")

    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Train and Validation R2 vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim(-.50, 1.0)

    plt.savefig(f"{args.output}.png", dpi=300)
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()