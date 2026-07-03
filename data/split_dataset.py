import os
import random
import shutil
from pathlib import Path
import hydra


@hydra.main(config_path=".", config_name="split_dataset")
def main(cfg):
    dataset_root = Path(cfg.dataset_root)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect all .npz files
    all_files = sorted(dataset_root.rglob("*.npz"))
    total_files = len(all_files)
    if total_files == 0:
        raise RuntimeError(f"No .npz files found in {dataset_root}")
    print(f"Found {total_files} files in {dataset_root}")

    random.seed(cfg.seed)
    random.shuffle(all_files)

    # Create dataset splits
    for split in cfg.splits:
        subset_size = max(1, int(total_files * split / 100))
        subset_files = all_files[:subset_size]

        subset_dir = output_root / f"{split}percent"
        subset_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating {split}% subset with {subset_size} files in {subset_dir}")

        for src in subset_files:
            dst = subset_dir / src.name
            shutil.copy(src, dst)

    print("Done!")

if __name__ == "__main__":
    main()