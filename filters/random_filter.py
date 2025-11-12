import random
import shutil
from pathlib import Path
from tqdm import tqdm


def keep_random_segments(root_dir: str, keep_count: int):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Root Directory does not exist: {root_path}")

    segment_parents = [p for p in root_path.rglob("*_segment") if p.is_dir()]

    if not segment_parents:
        print("No segment folders found")
        return

    delete_count = 0
    for parent in tqdm(segment_parents, desc="Processing Folders"):
        segments = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("segment_")]

        if len(segments) <= keep_count:
            continue

        keep_segments = random.sample(segments, keep_count)
        keep_names = {s.name for s in keep_segments}

        for seg in segments:
            if seg.name not in keep_names:
                shutil.rmtree(seg)
                delete_count += 1

    print(f"Deleted {delete_count} segments.")


if __name__ == "__main__":
    keep_random_segments(root_dir="data/wavs_20250416_012741_splits", keep_count=5)
