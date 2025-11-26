from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import List, Tuple
import click
from tqdm import tqdm


@dataclass
class CleanTask:
    """Defines a single cleaning task."""

    action: str  # 'clean_files', 'clean_folders', 'delete_by_extension', 'delete_empty_dirs'
    key_extension: str = ""
    query_extension: str = ""


class Cleaner:
    """Clean files and directories based on various criteria."""

    def __init__(self, root_dir: str, dry_run: bool = False):
        """
        Args:
            root_dir: Root directory to clean
            dry_run: If True, simulate actions without deleting
        """
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run

        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

    def _delete_file(self, file: Path) -> bool:
        """
        Delete a file.

        Returns:
            True if deleted (or would be in dry run), False if failed
        """
        try:
            if self.dry_run:
                logger.info(f"DRY RUN: Would delete {file}")
                return True
            else:
                file.unlink()
                logger.debug(f"Deleted: {file}")
                return True
        except Exception as e:
            logger.error(f"Couldn't delete {file}: {e}")
            return False

    def _delete_directory(self, directory: Path) -> bool:
        """
        Delete an empty directory.

        Returns:
            True if deleted (or would be in dry run), False if failed
        """
        try:
            if self.dry_run:
                logger.info(f"DRY RUN: Would delete directory {directory}")
                return True
            else:
                directory.rmdir()
                logger.debug(f"Deleted directory: {directory}")
                return True
        except Exception as e:
            logger.error(f"Couldn't delete directory {directory}: {e}")
            return False

    def clean_files(self, key_extension: str, query_extension: str) -> Tuple[int, int]:
        """
        Delete query files with no matching key file name.
        Example: delete '123.wav' if '123.json' does not exist.

        Args:
            key_extension: Extension that must exist (e.g., 'json')
            query_extension: Extension to delete if key missing (e.g., 'wav')

        Returns:
            Tuple of (deleted_count, total_count)
        """
        query_files = list(self.root_dir.rglob(f"*.{query_extension}"))
        deleted_count = 0

        for qf in tqdm(query_files, desc="Checking files", unit="files"):
            corresponding_file = qf.with_suffix(f".{key_extension}")
            if not corresponding_file.exists():
                if self._delete_file(qf):
                    deleted_count += 1

        logger.success(
            f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} "
            f"{deleted_count}/{len(query_files)} '{query_extension}' files "
            f"with no matching '.{key_extension}'"
        )

        return deleted_count, len(query_files)

    def clean_folders(
        self, key_extension: str, query_extension: str
    ) -> Tuple[int, int]:
        """
        Delete all query files inside any folder that contains zero key-extension files.

        Args:
            key_extension: Extension that must exist in folder (e.g., 'json')
            query_extension: Extension to delete if folder has no key files (e.g., 'wav')

        Returns:
            Tuple of (deleted_count, total_count)
        """
        query_files = list(self.root_dir.rglob(f"*.{query_extension}"))
        deleted_count = 0

        for qf in tqdm(query_files, desc="Checking folders", unit="files"):
            folder = qf.parent
            has_key_file = any(folder.glob(f"*.{key_extension}"))
            if not has_key_file:
                if self._delete_file(qf):
                    deleted_count += 1

        logger.success(
            f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} "
            f"{deleted_count}/{len(query_files)} '{query_extension}' files "
            f"from folders without '.{key_extension}' files"
        )

        return deleted_count, len(query_files)

    def delete_empty_dirs(self) -> int:
        """
        Delete empty subdirectories (bottom-up).

        Returns:
            Number of directories deleted
        """
        deleted_count = 0

        # Collect all directories sorted by depth (deepest first)
        all_dirs = sorted(
            [d for d in self.root_dir.rglob("*") if d.is_dir()],
            key=lambda x: len(x.parts),
            reverse=True,
        )

        for directory in tqdm(all_dirs, desc="Checking directories", unit="dirs"):
            # Skip root directory
            if directory == self.root_dir:
                continue

            # Check if directory is empty
            try:
                if not any(directory.iterdir()):
                    if self._delete_directory(directory):
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"Error checking directory {directory}: {e}")

        logger.success(
            f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} "
            f"{deleted_count} empty directories"
        )

        return deleted_count

    def delete_by_extension(self, extension: str) -> Tuple[int, int]:
        """
        Recursively delete all files with the given extension.

        Args:
            extension: File extension to delete (e.g., 'txt')

        Returns:
            Tuple of (deleted_count, total_count)
        """
        files = list(self.root_dir.rglob(f"*.{extension}"))
        deleted_count = 0

        for f in tqdm(files, desc=f"Deleting .{extension} files", unit="files"):
            if self._delete_file(f):
                deleted_count += 1

        logger.success(
            f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} "
            f"{deleted_count}/{len(files)} files with extension '.{extension}'"
        )

        return deleted_count, len(files)

    @classmethod
    def process_tasks(
        cls, root_dir: str, tasks: List[CleanTask], dry_run: bool = False
    ) -> dict:
        """
        Run multiple cleaning tasks efficiently.

        Args:
            root_dir: Root directory to clean
            tasks: List of CleanTask objects
            dry_run: If True, simulate without deleting

        Returns:
            Dictionary with statistics for each task
        """
        cleaner = cls(root_dir=root_dir, dry_run=dry_run)
        logger.info(f"Starting batch cleaning in {root_dir}")
        logger.info(f"Tasks to run: {len(tasks)}")
        if dry_run:
            logger.info("DRY RUN MODE - No files will be deleted")

        results = {}

        for i, task in enumerate(tasks, 1):
            logger.info(f"[{i}/{len(tasks)}] Running: {task.action}")

            if task.action == "clean_files":
                deleted, total = cleaner.clean_files(
                    task.key_extension, task.query_extension
                )
                results[f"{i}_{task.action}"] = {"deleted": deleted, "total": total}

            elif task.action == "clean_folders":
                deleted, total = cleaner.clean_folders(
                    task.key_extension, task.query_extension
                )
                results[f"{i}_{task.action}"] = {"deleted": deleted, "total": total}

            elif task.action == "delete_by_extension":
                deleted, total = cleaner.delete_by_extension(task.query_extension)
                results[f"{i}_{task.action}"] = {"deleted": deleted, "total": total}

            elif task.action == "delete_empty_dirs":
                deleted = cleaner.delete_empty_dirs()
                results[f"{i}_{task.action}"] = {"deleted": deleted}

            else:
                logger.warning(f"Unknown action: {task.action}")

        logger.success("Cleaning completed!")
        return results


# ============================================================================
# CLI with Click
# ============================================================================


@click.command()
@click.option("--root-dir", "-r", required=True, help="Root directory to clean")
@click.option(
    "--action",
    "-a",
    required=True,
    type=click.Choice(
        ["clean_files", "clean_folders", "delete_by_extension", "delete_empty_dirs"]
    ),
    help="Cleaning action to perform",
)
@click.option(
    "--key-ext", "-k", default="", help="Key extension (for clean_files/clean_folders)"
)
@click.option("--query-ext", "-q", default="", help="Query extension to delete")
@click.option("--dry-run", is_flag=True, help="Preview without deleting")
def main(root_dir, action, key_ext, query_ext, dry_run):
    """File and directory cleaning utility."""
    cleaner = Cleaner(root_dir=root_dir, dry_run=dry_run)

    if action == "clean_files":
        if not key_ext or not query_ext:
            raise click.UsageError("--key-ext and --query-ext required for clean_files")
        cleaner.clean_files(key_ext, query_ext)

    elif action == "clean_folders":
        if not key_ext or not query_ext:
            raise click.UsageError(
                "--key-ext and --query-ext required for clean_folders"
            )
        cleaner.clean_folders(key_ext, query_ext)

    elif action == "delete_by_extension":
        if not query_ext:
            raise click.UsageError("--query-ext required for delete_by_extension")
        cleaner.delete_by_extension(query_ext)

    elif action == "delete_empty_dirs":
        cleaner.delete_empty_dirs()


if __name__ == "__main__":
    main()
