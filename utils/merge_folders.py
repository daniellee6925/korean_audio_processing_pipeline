from pathlib import Path
from loguru import logger
from tqdm import tqdm
import shutil


class DirectoryMerger:

    def __init__(self, dir1: str, dir2: str, output_dir: str = None):
        """
        Merge two directories with the same structure.

        Args:
            dir1 (str): First directory path
            dir2 (str): Second directory path
            output_dir (str): Output directory. If None, merges into dir1
        """
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.output_dir = Path(output_dir) if output_dir else self.dir1

        if not self.dir1.exists():
            raise ValueError(f"Directory 1 does not exist: {self.dir1}")
        if not self.dir2.exists():
            raise ValueError(f"Directory 2 does not exist: {self.dir2}")

    def merge(self, copy_mode: bool = True) -> None:
        """
        Merge dir2 into the output directory.

        Args:
            copy_mode (bool): If True, copy files. If False, move files.
        """
        if self.output_dir != self.dir1:
            logger.info(f"Creating output directory: {self.output_dir}")
            if self.output_dir.exists():
                logger.warning(f"Output directory already exists: {self.output_dir}")
            else:
                logger.info(f"Copying {self.dir1} to {self.output_dir}")
                shutil.copytree(self.dir1, self.output_dir, dirs_exist_ok=True)

        all_files = list(self.dir2.rglob("*"))
        files_to_merge = [f for f in all_files if f.is_file()]

        logger.info(f"Found {len(files_to_merge)} files to merge from {self.dir2}")

        copied_count = 0
        skipped_count = 0

        for file in tqdm(files_to_merge, desc="Merging files"):
            relative_path = file.relative_to(self.dir2)
            dest_path = self.output_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if dest_path.exists():
                logger.warning(f"File already exists, skipping: {relative_path}")
                skipped_count += 1
                continue

            try:
                if copy_mode:
                    shutil.copy2(file, dest_path)
                    logger.debug(f"Copied: {relative_path}")
                else:
                    shutil.move(str(file), str(dest_path))
                    logger.debug(f"Moved: {relative_path}")
                copied_count += 1
            except Exception as e:
                logger.error(f"Failed to process {relative_path}: {e}")

        logger.info(f"Merge complete!")
        logger.info(f"Files copied/moved: {copied_count}")
        logger.info(f"Files skipped (already exist): {skipped_count}")
        logger.info(f"Output directory: {self.output_dir}")

    def merge_with_conflict_resolution(self, strategy: str = "skip") -> None:
        """
        Merge with conflict resolution strategy.

        Args:
            strategy (str): How to handle conflicts
                - "skip": Skip existing files (default)
                - "overwrite": Overwrite existing files
                - "rename": Rename new file with suffix
        """
        if self.output_dir != self.dir1:
            logger.info(f"Creating output directory: {self.output_dir}")
            if not self.output_dir.exists():
                shutil.copytree(self.dir1, self.output_dir, dirs_exist_ok=True)

        all_files = list(self.dir2.rglob("*"))
        files_to_merge = [f for f in all_files if f.is_file()]

        logger.info(f"Found {len(files_to_merge)} files to merge")
        logger.info(f"Conflict strategy: {strategy}")

        processed_count = 0

        for file in tqdm(files_to_merge, desc="Merging files"):
            relative_path = file.relative_to(self.dir2)
            dest_path = self.output_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if dest_path.exists():
                    if strategy == "skip":
                        logger.debug(f"Skipping existing: {relative_path}")
                        continue
                    elif strategy == "overwrite":
                        logger.debug(f"Overwriting: {relative_path}")
                        shutil.copy2(file, dest_path)
                    elif strategy == "rename":
                        # Find a new name
                        counter = 1
                        new_dest = dest_path
                        while new_dest.exists():
                            new_name = f"{dest_path.stem}_{counter}{dest_path.suffix}"
                            new_dest = dest_path.parent / new_name
                            counter += 1
                        logger.debug(f"Renaming to: {new_dest.name}")
                        shutil.copy2(file, new_dest)
                else:
                    shutil.copy2(file, dest_path)

                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {relative_path}: {e}")

        logger.info(f"Merge complete! Processed {processed_count} files")


if __name__ == "__main__":
    merger = DirectoryMerger(
        dir1="data/wavs_20250416_012741_splits_filtered",
        dir2="data/wavs_20250416_012741_splits_filtered_text",
        output_dir="merged",  # set if new directory needed
    )
    merger.merge(copy_mode=True)

    # merger.merge_with_conflict_resolution(strategy="rename")  # or "skip" or "overwrite"
