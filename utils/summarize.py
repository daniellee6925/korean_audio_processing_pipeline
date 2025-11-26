from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import soundfile as sf
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm
import click


@dataclass
class FileStats:
    """Statistics for a single audio file."""

    count: int = 0
    duration: float = 0.0
    size: int = 0


@dataclass
class Summary:
    """Complete directory summary."""

    root_directory: str
    total_files: int
    total_duration_seconds: float
    total_duration_formatted: str
    total_size_bytes: int
    total_size_formatted: str
    average_duration_seconds: float
    average_duration_formatted: str
    average_size_bytes: float
    average_size_formatted: str
    extension_breakdown: Dict[str, dict]


class AudioDirectorySummary:
    """Analyze audio files in a directory and generate statistics."""

    def __init__(
        self,
        root_dir: str,
        extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg"),
        save_json: bool = False,
        json_output: str = "audio_summary.json",
        max_workers: int = 8,
        deepest_only: bool = False,
    ):
        """
        Args:
            root_dir: Root directory to scan
            extensions: Audio file extensions to process
            save_json: Save summary to JSON file
            json_output: Output JSON filename
            max_workers: Number of parallel workers
            deepest_only: Only process files in leaf directories (no subdirs)
        """
        self.root_dir = Path(root_dir)
        self.extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
        )
        self.save_json = save_json
        self.json_output = json_output
        self.max_workers = max_workers
        self.deepest_only = deepest_only

        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

    @staticmethod
    def get_file_info(filepath: Path) -> Tuple[str, float, int]:
        """
        Extract duration and size from audio file.

        Returns:
            Tuple of (extension, duration_seconds, size_bytes)
        """
        try:
            info = sf.info(str(filepath))
            duration = info.duration
        except Exception as e:
            logger.warning(f"Could not get duration for {filepath.name}: {e}")
            duration = 0.0

        try:
            size = filepath.stat().st_size
        except Exception as e:
            logger.warning(f"Could not get size for {filepath.name}: {e}")
            size = 0

        return filepath.suffix.lower(), duration, size

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Convert seconds to human-readable duration (e.g., '2h30m45s')."""
        if seconds == 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return "".join(parts)

    @staticmethod
    def format_size(bytes_val: float) -> str:
        """Convert bytes to human-readable size (e.g., '1.23 GB')."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"

    def collect_files(self) -> List[Path]:
        """Collect all audio files to process."""
        if self.deepest_only:
            # Only process files in leaf directories (directories with no subdirectories)
            files = []
            for dirpath, subdirs, _ in self.root_dir.walk():
                if not subdirs:  # Leaf directory
                    files.extend(
                        [
                            f
                            for f in dirpath.iterdir()
                            if f.is_file() and f.suffix.lower() in self.extensions
                        ]
                    )
        else:
            # Process all matching files recursively
            files = [
                f
                for f in self.root_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in self.extensions
            ]

        return files

    def generate_summary(self) -> Summary:
        """
        Analyze all audio files and generate summary statistics.

        Returns:
            Summary dataclass with all statistics
        """
        logger.info(f"Scanning audio files in {self.root_dir}")

        # Collect files
        files = self.collect_files()
        total_files = len(files)

        if not files:
            logger.warning(f"No audio files found with extensions: {self.extensions}")
            return self._create_empty_summary()

        logger.info(f"Found {total_files} audio files")
        logger.info(f"Processing with {self.max_workers} workers...")

        # Process files in parallel with progress bar
        extension_stats = defaultdict(lambda: FileStats())
        total_duration = 0.0
        total_size = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.get_file_info, f): f for f in files}

            # Process results as they complete
            with tqdm(total=total_files, desc="Analyzing", unit="files") as pbar:
                for future in as_completed(future_to_file):
                    try:
                        ext, duration, size = future.result()

                        extension_stats[ext].count += 1
                        extension_stats[ext].duration += duration
                        extension_stats[ext].size += size

                        total_duration += duration
                        total_size += size

                    except Exception as e:
                        file_path = future_to_file[future]
                        logger.error(f"Error processing {file_path}: {e}")

                    pbar.update(1)

        # Calculate averages
        avg_duration = total_duration / total_files if total_files else 0
        avg_size = total_size / total_files if total_files else 0

        # Build summary
        summary = Summary(
            root_directory=str(self.root_dir.absolute()),
            total_files=total_files,
            total_duration_seconds=round(total_duration, 2),
            total_duration_formatted=self.format_duration(total_duration),
            total_size_bytes=total_size,
            total_size_formatted=self.format_size(total_size),
            average_duration_seconds=round(avg_duration, 2),
            average_duration_formatted=self.format_duration(avg_duration),
            average_size_bytes=round(avg_size, 2),
            average_size_formatted=self.format_size(avg_size),
            extension_breakdown={
                ext: {
                    "count": stats.count,
                    "duration_seconds": round(stats.duration, 2),
                    "duration_formatted": self.format_duration(stats.duration),
                    "size_bytes": stats.size,
                    "size_formatted": self.format_size(stats.size),
                }
                for ext, stats in sorted(extension_stats.items())
            },
        )

        # Save to JSON if requested
        if self.save_json:
            self._save_json(summary)

        # Print to console
        self.print_summary(summary)

        return summary

    def _create_empty_summary(self) -> Summary:
        """Create an empty summary when no files are found."""
        return Summary(
            root_directory=str(self.root_dir.absolute()),
            total_files=0,
            total_duration_seconds=0.0,
            total_duration_formatted="0s",
            total_size_bytes=0,
            total_size_formatted="0.00 B",
            average_duration_seconds=0.0,
            average_duration_formatted="0s",
            average_size_bytes=0.0,
            average_size_formatted="0.00 B",
            extension_breakdown={},
        )

    def _save_json(self, summary: Summary) -> None:
        """Save summary to JSON file."""
        try:
            output_path = Path(self.json_output)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved summary to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")

    @staticmethod
    def print_summary(summary: Summary) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 70)
        print("AUDIO DIRECTORY SUMMARY")
        print("=" * 70)
        print(f"Root Directory: {summary.root_directory}")
        print(f"Total Files: {summary.total_files:,}")
        print(
            f"Total Duration: {summary.total_duration_formatted} "
            f"({summary.total_duration_seconds:.2f} sec)"
        )
        print(
            f"Total Size: {summary.total_size_formatted} " f"({summary.total_size_bytes:,} bytes)"
        )
        print(
            f"Average Duration: {summary.average_duration_formatted} "
            f"({summary.average_duration_seconds:.2f} sec)"
        )
        print(
            f"Average File Size: {summary.average_size_formatted} "
            f"({summary.average_size_bytes:,.2f} bytes)"
        )

        if summary.extension_breakdown:
            print("\n" + "-" * 70)
            print("EXTENSION BREAKDOWN")
            print("-" * 70)
            for ext, stats in summary.extension_breakdown.items():
                print(f"\n{ext.upper()}:")
                print(f"  Files: {stats['count']:,}")
                print(
                    f"  Duration: {stats['duration_formatted']} "
                    f"({stats['duration_seconds']:.2f} sec)"
                )
                print(f"  Size: {stats['size_formatted']} " f"({stats['size_bytes']:,} bytes)")

        print("=" * 70 + "\n")


# ============================================================================
# CLI with Click
# ============================================================================


@click.command()
@click.option("--root-dir", "-r", required=True, help="Root directory to scan for audio files")
@click.option(
    "--extensions",
    "-e",
    default="wav,mp3,flac,m4a,ogg",
    help="Comma-separated audio extensions (default: wav,mp3,flac,m4a,ogg)",
)
@click.option("--save-json", "-j", is_flag=True, help="Save summary to JSON file")
@click.option(
    "--json-output",
    "-o",
    default="audio_summary.json",
    help="JSON output filename (default: audio_summary.json)",
)
@click.option(
    "--max-workers", "-w", default=8, type=int, help="Number of parallel workers (default: 8)"
)
@click.option(
    "--deepest-only", "-d", is_flag=True, help="Only process files in leaf directories (no subdirs)"
)
def main(root_dir, extensions, save_json, json_output, max_workers, deepest_only):
    """Analyze audio files in a directory and generate statistics."""

    # Parse extensions
    ext_list = [e.strip() for e in extensions.split(",")]

    summarizer = AudioDirectorySummary(
        root_dir=root_dir,
        extensions=tuple(ext_list),
        save_json=save_json,
        json_output=json_output,
        max_workers=max_workers,
        deepest_only=deepest_only,
    )

    summarizer.generate_summary()


if __name__ == "__main__":
    main()
