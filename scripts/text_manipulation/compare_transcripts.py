from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import os
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)


class TextFolderComparator:
    """
    Recursively compare paired txt files (v_segment_X.txt vs w_segment_X.txt)
    in subfolders and delete files where similarity of entire file falls below a threshold.
    """

    def __init__(
        self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS", threshold: float = 0.9
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def compare_pair(self, v_path: Path, w_path: Path) -> float:
        """
        Compare the full content of two text files and return cosine similarity.
        """
        with v_path.open("r", encoding="utf-8") as f:
            v_text = f.read().strip()

        with w_path.open("r", encoding="utf-8") as f:
            w_text = f.read().strip()

        embeddings_v = self.model.encode(v_text, convert_to_tensor=True)
        embeddings_w = self.model.encode(w_text, convert_to_tensor=True)

        similarity = util.cos_sim(embeddings_v, embeddings_w).item()
        return similarity

    def process_folder(self, root_dir: str, dry_run: bool = True):
        """
        Recursively process all subfolders and delete files below threshold.
        If dry_run=True, only prints counts and files to be deleted.
        """
        root_path = Path(root_dir)
        to_delete = []
        to_keep = []

        # Recursively find subfolders
        subfolders = [p for p in root_path.rglob("*") if p.is_dir()]
        print(f"Found {len(subfolders)} subfolders. Processing...")

        for subfolder in tqdm(subfolders, desc="Subfolders", unit="folder"):
            v_files = sorted(subfolder.glob("v_segment_*.txt"))
            w_files = sorted(subfolder.glob("w_segment_*.txt"))
            if not v_files or not w_files:
                continue

            for v_file, w_file in zip(v_files, w_files):

                sim = self.compare_pair(v_file, w_file)
                if sim < self.threshold:
                    to_delete.append((v_file, w_file, sim))
                else:
                    to_keep.append((v_file, w_file, sim))

        # Print summary
        print("=" * 50)
        print(f"Total segment pairs: {len(to_delete) + len(to_keep)}")
        print(f"Pairs below threshold ({self.threshold}): {len(to_delete)}")
        print(f"Pairs above threshold: {len(to_keep)}")
        if dry_run:
            print(f"This is a dry run")
        else:
            # Delete files
            for v_file, w_file, sim in to_delete:
                v_file.unlink()
                w_file.unlink()
            print(f"\nDeleted {len(to_delete)} segment pairs.")

        print("=" * 50)
        print("Done.")


# ----------------- USAGE ----------------- #

if __name__ == "__main__":
    comparator = TextFolderComparator(threshold=0.95)

    # Dry run first
    comparator.process_folder("Final_trans", dry_run=False)

    # Confirm deletion
    # comparator.process_folder("Final_trans", dry_run=False)
