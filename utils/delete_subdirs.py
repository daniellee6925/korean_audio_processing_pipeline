from pathlib import Path
from loguru import logger
from tqdm import tqdm
import shutil


class DeleteSegmentSubdirs:
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): Root folder containing all TAKE folders
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {root_dir}")

    def delete_subdirs_in_segments(self):
        """
        Deletes all subdirectories inside segment folders (segment_*) at the specified depth.
        Keeps files in segment_* folders intact.
        """
        # Iterate over all TAKE folders
        take_folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        removed_count = 0
        for take_folder in tqdm(take_folders, desc="Processing TAKE folders"):
            # Find all segment folders recursively
            segment_folders = [p for p in take_folder.rglob("segment_*") if p.is_dir()]
            for segment in segment_folders:
                # Delete only subdirectories inside the segment folder
                for subdir in [d for d in segment.iterdir() if d.is_dir()]:
                    try:
                        shutil.rmtree(subdir)
                        removed_count += 1
                        logger.info(f"Removed subdir: {subdir}")
                    except Exception as e:
                        logger.warning(f"Could not remove {subdir}: {e}")

        logger.info(f"Finished. Total subdirectories removed: {removed_count}")


if __name__ == "__main__":
    cleaner = DeleteSegmentSubdirs(root_dir="data/wavs_20250416_012741_splits copy")
    cleaner.delete_subdirs_in_segments()
