import random
import shutil
from pathlib import Path
from tqdm import tqdm


def keep_random_segments(root_dir: str, keep_count: int, keep_type: "dir"):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Root Directory does not exist: {root_path}")

    segment_parents = [p for p in root_path.rglob("*_segment") if p.is_dir()]

    if not segment_parents:
        print("No segment folders found")
        return

    delete_count = 0
    for parent in tqdm(segment_parents, desc="Processing Folders"):
        if keep_type == "dir":
            segments = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("segment_")]
        elif keep_type == "csv":
            segments = [
                p
                for p in parent.iterdir()
                if p.is_file() and p.suffix == ".csv" and p.name.startswith("segment_")
            ]
        elif keep_type == "wav":
            segments = [
                p
                for p in parent.iterdir()
                if p.is_file() and p.suffix == ".wav" and p.name.startswith("segment_")
            ]
        elif keep_type == "all":
            segments = [p for p in parent.iterdir() if p.name.startswith("segment_")]
        else:
            raise ValueError(f"Invalid keep_type: {keep_type}")

        if len(segments) <= keep_count:
            continue

        keep_segments = random.sample(segments, keep_count)
        keep_names = {s.name for s in keep_segments}

        for seg in segments:
            if seg.name not in keep_names:
                if seg.is_dir():
                    shutil.rmtree(seg)
                else:
                    seg.unlink()
                delete_count += 1

    print(f"Deleted {delete_count} segments.")


if __name__ == "__main__":
    keep_random_segments(root_dir="data/wavs_20250416_012741_splits", keep_count=5)
