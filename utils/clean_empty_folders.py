from pathlib import Path
from loguru import logger
from tqdm import tqdm


class CleanEmptyFolders:

    def __init__(self, root_dir: str, extension: str = "wav"):
        """
        Args:
            root_dir (str): Root directory to clean.
            extension (str): Audio file extension to keep (e.g., "wav").
        """
        self.root_dir = Path(root_dir)
        self.audio_extension = f".{extension.lower()}"

        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")

    def get_depth(self, path: Path) -> int:
        """Calculate the depth of a path relative to root_dir."""
        try:
            relative = path.relative_to(self.root_dir)
            return len(relative.parts)
        except ValueError:
            return -1

    def find_max_depth(self) -> int:
        """Find the maximum depth of all folders in the directory tree."""
        max_depth = 0
        for item in self.root_dir.rglob("*"):
            if item.is_dir():
                depth = self.get_depth(item)
                max_depth = max(max_depth, depth)
        return max_depth

    def has_any_files(self, folder: Path) -> bool:
        """Check if a folder contains any files (at any level)."""
        return any(item.is_file() for item in folder.rglob("*"))

    def delete_folders_without_files(self) -> None:
        """
        Delete all folders that do not contain any files at the end (deepest level).
        Works from deepest to shallowest to handle nested empty folders.
        """
        max_depth = self.find_max_depth()

        if max_depth == 0:
            logger.warning("No subfolders found in root directory")
            return

        logger.info(f"Scanning for folders without files (max depth: {max_depth})")

        removed_count = 0

        # Process from deepest to shallowest to handle cascading deletions
        for depth in range(max_depth, 0, -1):
            all_folders = [f for f in self.root_dir.rglob("*") if f.is_dir()]
            folders_at_depth = [f for f in all_folders if self.get_depth(f) == depth]

            for folder in tqdm(folders_at_depth, desc=f"Checking folders at depth {depth}"):
                if not self.has_any_files(folder):
                    try:
                        # Remove all subdirectories
                        for subdir in sorted(folder.rglob("*"), reverse=True):
                            if subdir.is_dir():
                                try:
                                    subdir.rmdir()
                                except OSError:
                                    pass

                        # Remove the folder itself
                        folder.rmdir()
                        removed_count += 1
                        logger.info(f"Removed empty folder: {folder}")
                    except OSError as e:
                        logger.warning(f"Could not remove {folder}: {e}")

        logger.info(f"Finished. Total empty folders removed: {removed_count}")

    def delete_files(self) -> None:
        """
        Automatically detect the deepest level and delete folders at that level
        that contain no audio files with the allowed extension.
        """
        # Automatically find the deepest level
        target_depth = self.find_max_depth()

        if target_depth == 0:
            logger.warning("No subfolders found in root directory")
            return

        logger.info(f"Automatically detected target depth: {target_depth}")

        removed_count = 0

        # Get all folders at the target depth
        all_folders = [f for f in self.root_dir.rglob("*") if f.is_dir()]
        target_folders = [f for f in all_folders if self.get_depth(f) == target_depth]

        logger.info(f"Found {len(target_folders)} folders at depth {target_depth}")

        for folder in tqdm(target_folders, desc=f"Cleaning folders at depth {target_depth}"):
            has_audio = any(
                f.is_file() and f.suffix.lower() == self.audio_extension for f in folder.rglob("*")
            )

            if has_audio:
                continue
            try:
                for item in folder.rglob("*"):
                    if item.is_file():
                        item.unlink()

                for subdir in sorted(folder.rglob("*"), reverse=True):
                    if subdir.is_dir():
                        try:
                            subdir.rmdir()
                        except OSError:
                            pass

                folder.rmdir()
                removed_count += 1
                logger.info(f"Removed: {folder}")
            except OSError as e:
                logger.warning(f"Could not remove {folder}: {e}")

        logger.info(
            f"Finished cleaning. Total folders removed at depth {target_depth}: {removed_count}"
        )

    def process_all(self):
        self.delete_files()
        self.delete_folders_without_files()


if __name__ == "__main__":
    cleaner = CleanEmptyFolders(root_dir="audio_files_1_sentences_trans", extension="wav")

    # cleaner.process_all()

    cleaner.delete_folders_without_files()
