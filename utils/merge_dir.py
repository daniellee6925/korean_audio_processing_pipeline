from pathlib import Path
from loguru import logger
from tqdm import tqdm
import shutil
import click


class DirectoryMerger:
    """Merge two directories with the same structure."""

    def __init__(
        self, dir1: str, dir2: str, output_dir: str = None, dry_run: bool = False
    ):
        """
        Args:
            dir1: First directory path
            dir2: Second directory path
            output_dir: Output directory. If None, merges into dir1
            dry_run: If True, preview without copying/moving files
        """
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.output_dir = Path(output_dir) if output_dir else self.dir1
        self.dry_run = dry_run

        if not self.dir1.exists():
            raise ValueError(f"Directory 1 does not exist: {self.dir1}")
        if not self.dir2.exists():
            raise ValueError(f"Directory 2 does not exist: {self.dir2}")

    def merge(self, copy_mode: bool = True, conflict_strategy: str = "skip") -> dict:
        """
        Merge dir2 into the output directory.

        Args:
            copy_mode: If True, copy files. If False, move files
            conflict_strategy: How to handle conflicts: "skip", "overwrite", or "rename"

        Returns:
            Dictionary with merge statistics
        """
        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be copied/moved")

        # Create output directory if different from dir1
        if self.output_dir != self.dir1:
            logger.info(f"Output directory: {self.output_dir}")
            if self.output_dir.exists():
                logger.warning(f"Output directory already exists: {self.output_dir}")
            else:
                if not self.dry_run:
                    logger.info(f"Copying {self.dir1} to {self.output_dir}")
                    shutil.copytree(self.dir1, self.output_dir, dirs_exist_ok=True)
                else:
                    logger.info(f"DRY RUN: Would copy {self.dir1} to {self.output_dir}")

        # Collect files to merge
        all_files = list(self.dir2.rglob("*"))
        files_to_merge = [f for f in all_files if f.is_file()]

        logger.info(f"Found {len(files_to_merge)} files to merge from {self.dir2}")
        logger.info(f"Copy mode: {'copy' if copy_mode else 'move'}")
        logger.info(f"Conflict strategy: {conflict_strategy}")

        stats = {
            "processed": 0,
            "skipped": 0,
            "renamed": 0,
            "overwritten": 0,
            "failed": 0,
        }

        for file in tqdm(files_to_merge, desc="Merging files", unit="files"):
            relative_path = file.relative_to(self.dir2)
            dest_path = self.output_dir / relative_path

            try:
                # Create parent directory
                if not self.dry_run:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Handle conflicts
                if dest_path.exists():
                    if conflict_strategy == "skip":
                        logger.debug(f"Skipping existing: {relative_path}")
                        stats["skipped"] += 1
                        continue
                    elif conflict_strategy == "overwrite":
                        logger.debug(f"Overwriting: {relative_path}")
                        if not self.dry_run:
                            if copy_mode:
                                shutil.copy2(file, dest_path)
                            else:
                                shutil.move(str(file), str(dest_path))
                        stats["overwritten"] += 1
                        stats["processed"] += 1
                        continue
                    elif conflict_strategy == "rename":
                        # Find a new name
                        counter = 1
                        new_dest = dest_path
                        while new_dest.exists():
                            new_name = f"{dest_path.stem}_{counter}{dest_path.suffix}"
                            new_dest = dest_path.parent / new_name
                            counter += 1
                        logger.debug(f"Renaming to: {new_dest.name}")
                        dest_path = new_dest
                        stats["renamed"] += 1

                # Copy or move file
                if not self.dry_run:
                    if copy_mode:
                        shutil.copy2(file, dest_path)
                        logger.debug(f"Copied: {relative_path}")
                    else:
                        shutil.move(str(file), str(dest_path))
                        logger.debug(f"Moved: {relative_path}")
                else:
                    action = "copy" if copy_mode else "move"
                    logger.debug(f"DRY RUN: Would {action}: {relative_path}")

                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Failed to process {relative_path}: {e}")
                stats["failed"] += 1

        # Print summary
        logger.info("=" * 70)
        logger.info("MERGE COMPLETE!")
        logger.info(f"Files processed: {stats['processed']}")
        logger.info(f"Files skipped (already exist): {stats['skipped']}")
        if stats["renamed"] > 0:
            logger.info(f"Files renamed (conflicts): {stats['renamed']}")
        if stats["overwritten"] > 0:
            logger.info(f"Files overwritten: {stats['overwritten']}")
        if stats["failed"] > 0:
            logger.info(f"Files failed: {stats['failed']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70)

        return stats


# ============================================================================
# CLI with Click
# ============================================================================


@click.command()
@click.option("--dir1", "-d1", required=True, help="First directory path")
@click.option("--dir2", "-d2", required=True, help="Second directory path")
@click.option(
    "--output", "-o", default=None, help="Output directory (default: merge into dir1)"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["copy", "move"]),
    default="copy",
    help="Copy or move files (default: copy)",
)
@click.option(
    "--conflict",
    "-c",
    type=click.Choice(["skip", "overwrite", "rename"]),
    default="skip",
    help="How to handle conflicts (default: skip)",
)
@click.option("--dry-run", is_flag=True, help="Preview without copying/moving files")
def main(dir1, dir2, output, mode, conflict, dry_run):
    """Merge two directories with the same structure."""

    merger = DirectoryMerger(dir1=dir1, dir2=dir2, output_dir=output, dry_run=dry_run)

    merger.merge(copy_mode=(mode == "copy"), conflict_strategy=conflict)


if __name__ == "__main__":
    main()
