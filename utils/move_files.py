import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import shutil


class TxtFileOrganizer:
    def __init__(self, txt_root: str, folder_root: str):
        """
        Args:
            txt_root (str): Path containing all .txt files
            folder_root (str): Path containing all target folders
        """
        self.txt_root = Path(txt_root)
        self.folder_root = Path(folder_root)

        if not self.txt_root.exists():
            raise ValueError(f"TXT root does not exist: {txt_root}")
        if not self.folder_root.exists():
            raise ValueError(f"Folder root does not exist: {folder_root}")

    def organize(self):
        txt_files = list(self.txt_root.glob("*.txt"))
        if not txt_files:
            logger.warning(f"No .txt files found in {self.txt_root}")
            return

        moved_count = 0
        for txt_file in tqdm(txt_files, desc="Organizing TXT files"):
            # Extract prefix (e.g., 180101_003 from 180101_003_Tr1.txt)
            prefix = txt_file.stem.split("_Tr")[0]

            # Find folder starting with prefix
            target_folder = None
            for folder in self.folder_root.iterdir():
                if folder.is_dir() and folder.name.startswith(prefix):
                    target_folder = folder
                    break

            if target_folder:
                dest = target_folder / txt_file.name
                shutil.move(str(txt_file), str(dest))
                moved_count += 1
                logger.info(f"Moved {txt_file.name} → {target_folder}")
            else:
                logger.warning(f"No matching folder found for {txt_file.name}")

        logger.info(f"Finished organizing. Total files moved: {moved_count}")


if __name__ == "__main__":
    organizer = TxtFileOrganizer(
        txt_root="TEXT_251107",  # Folder where your .txt files currently are
        folder_root="1107_Recording_sentences",  # Folder containing TAKE folders
    )
    organizer.organize()
