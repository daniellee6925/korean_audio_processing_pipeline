from pathlib import Path
from loguru import logger
from typing import Union, List, Optional
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass
import time
import click


class RenameStatus(Enum):
    SUCCESS = "success"
    NO_CHANGE = "no_change"
    COLLISION = "collision"
    ERROR = "error"


@dataclass
class RenameResult:
    status: RenameStatus
    old_path: Path
    new_path: Optional[Path] = None
    error: Optional[str] = None


class ChangeFileName:
    def __init__(
        self,
        root_dir: str,
        extensions: Union[str, List[str]],
        original: str,
        change_to: str,
        portion: str = "any",
        custom_range: tuple[int, int] = None,
        dry_run: bool = False,
        suffix_digits: int = 3,
        target: str = "files",  # "files", "folders", or "both"
    ):
        self.root_dir = Path(root_dir)
        self.extensions = [extensions] if isinstance(extensions, str) else extensions
        self.original = original
        self.change_to = change_to
        self.portion = portion
        self.custom_range = custom_range
        self.dry_run = dry_run
        self.suffix_digits = suffix_digits
        self.target = target

        if not self.root_dir.exists():
            raise ValueError(f"Folder does not exist: {root_dir}")

        if target not in ["files", "folders", "both"]:
            raise ValueError(f"target must be 'files', 'folders', or 'both', got: {target}")

    def _suffix(self, name: str) -> str:
        """Extract and zero-pad suffix after last underscore"""
        if "_" not in name:
            return name
        prefix, suffix = name.rsplit("_", 1)
        return f"{prefix}_{suffix[-self.suffix_digits:].zfill(self.suffix_digits)}"

    def _replace_start(self, name: str) -> str:
        """Replace at start of name"""
        if name.startswith(self.original):
            return self.change_to + name[len(self.original) :]
        return name

    def _replace_end(self, name: str) -> str:
        """Replace at end of name"""
        if name.endswith(self.original):
            return name[: -len(self.original)] + self.change_to
        return name

    def _replace_any(self, name: str) -> str:
        """Replace anywhere in name"""
        return name.replace(self.original, self.change_to)

    def _replace_custom(self, name: str) -> str:
        """Replace within custom range"""
        start, end = self.custom_range
        portion = name[start:end]
        return name[:start] + portion.replace(self.original, self.change_to) + name[end:]

    def transform(self, name: str) -> str:
        """Transform filename according to specified portion"""
        if self.portion == "suffix":
            return self._suffix(name)
        elif self.portion == "start":
            return self._replace_start(name)
        elif self.portion == "end":
            return self._replace_end(name)
        elif self.portion == "custom":
            if not self.custom_range:
                raise ValueError("custom_range required for portion='custom'")
            return self._replace_custom(name)
        # default: any
        return self._replace_any(name)

    def rename_item(self, item_path: Path, is_folder: bool = False) -> RenameResult:
        """
        Rename a single file or folder.

        Args:
            item_path: Path to the file or folder
            is_folder: Whether this is a folder (affects name extraction)

        Returns:
            RenameResult with status and paths
        """
        try:
            # Extract name (use .name for folders, .stem for files)
            old_name = item_path.name if is_folder else item_path.stem
            new_name = self.transform(old_name)

            # No change needed
            if new_name == old_name:
                return RenameResult(status=RenameStatus.NO_CHANGE, old_path=item_path)

            # Build new path
            new_path = (
                item_path.with_name(new_name)
                if is_folder
                else item_path.with_name(new_name + item_path.suffix)
            )

            # Name collision
            if new_path.exists():
                item_type = "folder" if is_folder else "file"
                logger.warning(f"SKIPPED (collision): {item_path.name} -> {new_path.name}")
                return RenameResult(
                    status=RenameStatus.COLLISION,
                    old_path=item_path,
                    new_path=new_path,
                    error=f"Target {item_type} already exists",
                )

            # Perform rename
            if not self.dry_run:
                item_path.rename(new_path)
                logger.debug(f"Renamed: {item_path.name} -> {new_path.name}")
            else:
                prefix = "DRY RUN (folder)" if is_folder else "DRY RUN"
                logger.info(f"{prefix}: {item_path.name} -> {new_path.name}")

            return RenameResult(status=RenameStatus.SUCCESS, old_path=item_path, new_path=new_path)

        except Exception as e:
            logger.error(f"Error renaming {item_path}: {e}")
            return RenameResult(status=RenameStatus.ERROR, old_path=item_path, error=str(e))

    def preview_changes(self, limit: int = 10) -> None:
        """Preview first N renames without executing"""
        files = (
            [f for ext in self.extensions for f in self.root_dir.rglob(f"*.{ext}")]
            if self.target in ["files", "both"]
            else []
        )
        folders = (
            sorted(list(self.root_dir.rglob("*/")), key=lambda x: len(x.parts))
            if self.target in ["folders", "both"]
            else []
        )

        total_items = len(files) + len(folders)
        if not total_items:
            target_desc = {"files": "files", "folders": "folders", "both": "files or folders"}[
                self.target
            ]
            logger.info(f"No {target_desc} found under {self.root_dir}")
            return

        logger.info(f"Preview of first {min(limit, total_items)} renames:")
        logger.info(f"  Files: {len(files)}, Folders: {len(folders)}, Total: {total_items}")
        logger.info("-" * 70)

        change_count = 0
        for item, is_folder in [(f, True) for f in folders[:limit]] + [
            (f, False) for f in files[: limit - len(folders[:limit])]
        ]:
            old_name = item.name if is_folder else item.stem
            new_name = self.transform(old_name)
            prefix = "[DIR] " if is_folder else ""
            suffix = " (no change)" if new_name == old_name else ""
            separator = "" if suffix else " → "

            logger.info(f"  {prefix}{old_name}{separator}{new_name if not suffix else ''}{suffix}")
            if new_name != old_name:
                change_count += 1

        logger.info("-" * 70)
        logger.info(f"Items with changes: {change_count}/{min(limit, total_items)}")
        if total_items > limit:
            logger.info(f"... and {total_items - limit} more items not shown")

    def process_all(self) -> dict:
        """
        Process all files and/or folders sequentially.
        Folders are processed in reverse depth order to avoid path issues.

        Returns:
            Dictionary with processing statistics
        """
        # Collect items to process
        files = (
            [f for ext in self.extensions for f in self.root_dir.rglob(f"*.{ext}")]
            if self.target in ["files", "both"]
            else []
        )
        folders = (
            sorted(
                [d for d in self.root_dir.rglob("*") if d.is_dir()],
                key=lambda x: len(x.parts),
                reverse=True,
            )
            if self.target in ["folders", "both"]
            else []
        )

        total_items = len(files) + len(folders)
        if not total_items:
            target_desc = {"files": "files", "folders": "folders", "both": "files or folders"}[
                self.target
            ]
            logger.info(f"No {target_desc} found under {self.root_dir}")
            return {}

        logger.info(f"Found {len(files)} files and {len(folders)} folders to process")
        if self.dry_run:
            logger.info("DRY RUN MODE - No items will be modified")

        # Statistics
        stats = {
            "total": 0,
            "success": 0,
            "no_change": 0,
            "collision": 0,
            "error": 0,
            "files_processed": 0,
            "folders_processed": 0,
        }
        start_time = time.time()

        # Helper to process item and update stats
        def process_item(item_path: Path, is_folder: bool):
            try:
                result = self.rename_item(item_path, is_folder)
                stats["total"] += 1
                stats["folders_processed" if is_folder else "files_processed"] += 1
                stats[result.status.value] += 1
            except Exception as e:
                logger.exception(f"Unexpected exception processing {item_path}: {e}")
                stats["error"] += 1
                stats["total"] += 1
                stats["folders_processed" if is_folder else "files_processed"] += 1

        try:
            with tqdm(total=total_items, desc="Processing", unit="items") as pbar:
                # Process folders first (deepest first to avoid path issues)
                for folder_path in folders:
                    process_item(folder_path, is_folder=True)
                    pbar.update(1)

                # Process files
                for file_path in files:
                    process_item(file_path, is_folder=False)
                    pbar.update(1)

        except KeyboardInterrupt:
            logger.warning("\nKeyboardInterrupt received! Stopping...")

        finally:
            elapsed = time.time() - start_time
            avg_speed = stats["total"] / elapsed if elapsed > 0 else 0

            # Print summary
            logger.info("=" * 70)
            logger.info("FINISHED!")
            logger.info(f"Total items processed: {stats['total']}/{total_items}")
            logger.info(
                f"  Files: {stats['files_processed']}, Folders: {stats['folders_processed']}"
            )
            logger.info(f"  ✓ Successful renames: {stats['success']}")
            logger.info(f"  → No change needed: {stats['no_change']}")
            logger.info(f"  ! Name collisions: {stats['collision']}")
            logger.info(f"  ✗ Errors: {stats['error']}")
            logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
            logger.info(f"Average speed: {avg_speed:.2f} items/second")
            logger.info("=" * 70)

        return stats


# ============================================================================
# CLI with Click
# ============================================================================


@click.command()
@click.option("--root-dir", "-r", required=True, help="Root Directory")
@click.option("--extensions", "-e", default="wav", help="File extension (e.g., wav)")
@click.option("--original", "-o", required=True, help="Text to replace")
@click.option("--change-to", "-c", required=True, help="New text")
@click.option("--portion", "-p", default="any", help="Where: any/start/end")
@click.option("--dry-run", is_flag=True, help="Preview only")
@click.option(
    "--target",
    "-t",
    default="files",
    type=click.Choice(["files", "folders", "both"]),
    help="What to rename: files, folders, or both",
)
def main(root_dir, extensions, original, change_to, portion, dry_run, target):
    """simple file and folder renamer"""
    renamer = ChangeFileName(
        root_dir=root_dir,
        extensions=extensions,
        original=original,
        change_to=change_to,
        portion=portion,
        dry_run=dry_run,
        target=target,
    )
    renamer.process_all()


if __name__ == "__main__":
    main()
