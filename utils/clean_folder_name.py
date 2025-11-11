from pathlib import Path
from loguru import logger


class CleanFolderName:
    def __init__(self, root_dir: str, original: str, change_to: str):
        self.root_dir = Path(root_dir)
        self.original = original
        self.change_to = change_to

        if not self.root_dir.exists():
            raise ValueError(f"Folder does not exist: {root_dir}")

    def clean_folder_name(self, folder_path: Path):
        """Rename a single folder if its name contains the target substring."""
        if self.original in folder_path.name:
            new_name = folder_path.name.replace(self.original, self.change_to)
            new_path = folder_path.with_name(new_name)
            folder_path.rename(new_path)
            logger.info(f"Renamed folder: {folder_path} → {new_path}")
            return new_path
        return folder_path

    def process_all(self):
        """Recursively process all subfolders under the root directory."""
        # Sort by depth (deepest first) to avoid renaming parent dirs before children
        all_folders = sorted(
            [p for p in self.root_dir.rglob("*") if p.is_dir()],
            key=lambda x: len(x.parts),
            reverse=True,
        )

        if not all_folders:
            logger.info(f"No folders found under {self.root_dir}")
            return

        for folder in all_folders:
            self.clean_folder_name(folder)

        logger.info(f"Finished processing {len(all_folders)} folders.")


if __name__ == "__main__":
    cleaner = CleanFolderName(
        root_dir="TEXT_251107",
        original="_Tr1_segments",
        change_to=".TAKE",
    )
    cleaner.process_all()
