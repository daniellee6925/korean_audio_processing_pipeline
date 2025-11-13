from pathlib import Path


def get_folder_count(root_dir: str):
    root_path = Path(root_dir)

    if not root_path.exists():
        raise ValueError(f"Root Directory does not exist: {root_path}")

    all_folder = [p for p in root_path.glob("*") if p.is_dir()]
    count = len(all_folder)

    print(f"Found {count} folders in {root_path}")
    return count


if __name__ == "__main__":
    get_folder_count(root_dir="data/wavs_20250416_012741_splits_filtered")
