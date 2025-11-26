from pathlib import Path
from typing import List
import re
import click
from loguru import logger
from tqdm import tqdm


class TextSplitter:
    """Split text files using configurable patterns and save segments."""

    SPLIT_PATTERNS = {
        "newline": r"\n+",
        "comma": r",",
        "period": r"\.",
        "semicolon": r";",
        "pipe": r"\|",
        "tab": r"\t",
        "space": r"\s+",
        "custom": None,  # Will be provided by user
    }

    def __init__(
        self,
        root_dir: str,
        output_suffix: str = "_segments",
        split_by: str = "newline",
        custom_pattern: str = None,
        line_folders: bool = True,
        dry_run: bool = False,
    ):
        """
        Args:
            root_dir: Root directory to recursively scan for text files
            output_suffix: Suffix added to create segment folder for each text file
            split_by: Split pattern type (newline, comma, period, etc.)
            custom_pattern: Custom regex pattern when split_by="custom"
            line_folders: If True, each segment stored in its own subfolder
            dry_run: If True, preview without creating files
        """
        self.root_dir = Path(root_dir)
        self.output_suffix = output_suffix
        self.split_by = split_by
        self.line_folders = line_folders
        self.dry_run = dry_run

        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        # Set split pattern
        if split_by == "custom":
            if not custom_pattern:
                raise ValueError("custom_pattern required when split_by='custom'")
            self.pattern = custom_pattern
        else:
            if split_by not in self.SPLIT_PATTERNS:
                raise ValueError(
                    f"Invalid split_by. Choose from: {list(self.SPLIT_PATTERNS.keys())}"
                )
            self.pattern = self.SPLIT_PATTERNS[split_by]

    def split_text(self, text: str) -> List[str]:
        """Split text using the configured pattern."""
        segments = re.split(self.pattern, text)
        return [seg.strip() for seg in segments if seg.strip()]

    def process_text_file(self, file_path: Path) -> int:
        """
        Split a single text file and save each segment.

        Returns:
            Number of segments created
        """
        try:
            text = file_path.read_text(encoding="utf-8")
            segments = self.split_text(text)

            if not segments:
                logger.warning(f"No segments found in {file_path}")
                return 0

            # Create output directory
            output_dir = file_path.parent / f"{file_path.stem}{self.output_suffix}"

            if not self.dry_run:
                output_dir.mkdir(exist_ok=True)
            else:
                logger.info(f"DRY RUN: Would create {output_dir}")

            # Save segments
            for i, segment in enumerate(segments, start=1):
                if self.line_folders:
                    segment_dir = output_dir / f"segment_{i:04d}"
                    segment_file = segment_dir / f"segment_{i:04d}.txt"

                    if not self.dry_run:
                        segment_dir.mkdir(exist_ok=True)
                        segment_file.write_text(segment, encoding="utf-8")
                    else:
                        logger.debug(f"DRY RUN: Would create {segment_file}")
                else:
                    segment_file = output_dir / f"segment_{i:04d}.txt"

                    if not self.dry_run:
                        segment_file.write_text(segment, encoding="utf-8")
                    else:
                        logger.debug(f"DRY RUN: Would create {segment_file}")

            if not self.dry_run:
                logger.debug(f"Processed {file_path.name}: {len(segments)} segments")

            return len(segments)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return 0

    def process_directory(self) -> dict:
        """
        Recursively process all text files under the root directory.

        Returns:
            Dictionary with processing statistics
        """
        # Collect all text files
        text_files = list(self.root_dir.rglob("*.txt"))

        if not text_files:
            logger.info(f"No .txt files found in {self.root_dir}")
            return {"files_processed": 0, "total_segments": 0}

        logger.info(f"Found {len(text_files)} text files to process")
        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be created")

        stats = {"files_processed": 0, "total_segments": 0, "files_skipped": 0}

        with tqdm(total=len(text_files), desc="Processing", unit="files") as pbar:
            for file_path in text_files:
                segment_count = self.process_text_file(file_path)

                if segment_count > 0:
                    stats["files_processed"] += 1
                    stats["total_segments"] += segment_count
                else:
                    stats["files_skipped"] += 1

                pbar.update(1)

        # Print summary
        logger.info("=" * 70)
        logger.info("FINISHED!")
        logger.info(f"Files processed: {stats['files_processed']}/{len(text_files)}")
        logger.info(f"Total segments created: {stats['total_segments']}")
        logger.info(f"Files skipped (no segments): {stats['files_skipped']}")
        logger.info("=" * 70)

        return stats

    def preview(self, limit: int = 3) -> None:
        """Preview what would be created for the first N files."""
        text_files = list(self.root_dir.rglob("*.txt"))[:limit]

        if not text_files:
            logger.info(f"No .txt files found in {self.root_dir}")
            return

        logger.info(f"Preview of first {len(text_files)} files:")
        logger.info(f"Split pattern: {self.split_by} -> {self.pattern}")
        logger.info("-" * 70)

        for file_path in text_files:
            try:
                text = file_path.read_text(encoding="utf-8")
                segments = self.split_text(text)

                logger.info(f"\n{file_path.name}:")
                logger.info(f"  → {len(segments)} segments")
                logger.info(f"  → Output: {file_path.stem}{self.output_suffix}/")

                # Show first few segments
                for i, seg in enumerate(segments[:3], start=1):
                    preview = seg[:60] + "..." if len(seg) > 60 else seg
                    logger.info(f"     Segment {i}: {preview}")

                if len(segments) > 3:
                    logger.info(f"     ... and {len(segments) - 3} more segments")

            except Exception as e:
                logger.error(f"Error previewing {file_path}: {e}")

        logger.info("-" * 70)


# ============================================================================
# CLI with Click
# ============================================================================


@click.command()
@click.option("--root-dir", "-r", required=True, help="Root directory containing text files")
@click.option(
    "--output-suffix",
    "-s",
    default="_segments",
    help="Suffix for output folders (default: _segments)",
)
@click.option(
    "--split-by",
    "-b",
    type=click.Choice(
        ["newline", "comma", "period", "semicolon", "pipe", "tab", "space", "custom"]
    ),
    default="newline",
    help="How to split text (default: newline)",
)
@click.option(
    "--custom-pattern", "-p", default=None, help="Custom regex pattern when --split-by=custom"
)
@click.option(
    "--line-folders/--no-line-folders",
    "-f/-F",
    default=True,
    help="Create subfolder for each segment (default: True)",
)
@click.option("--dry-run", is_flag=True, help="Preview without creating files")
@click.option("--preview", is_flag=True, help="Show preview of first 3 files and exit")
def main(root_dir, output_suffix, split_by, custom_pattern, line_folders, dry_run, preview):
    """Split text files using configurable patterns."""

    splitter = TextSplitter(
        root_dir=root_dir,
        output_suffix=output_suffix,
        split_by=split_by,
        custom_pattern=custom_pattern,
        line_folders=line_folders,
        dry_run=dry_run,
    )

    if preview:
        splitter.preview()
    else:
        splitter.process_directory()


if __name__ == "__main__":
    main()
