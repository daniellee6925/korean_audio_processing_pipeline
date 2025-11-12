import os
import librosa
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import soundfile as sf
from loguru import logger


class AudioDirectorySummary:
    """
    Generates summary statistics for audio files in a directory structure.
    """

    def __init__(
        self,
        root_dir: str,
        extensions: tuple = (".wav"),
        save_json: bool = False,
        json_output: str = "audio_summary.json",
        max_workers: int = 8,
    ):
        self.root_dir = root_dir
        self.extensions = extensions
        self.save_json = save_json
        self.json_output = json_output
        self.max_workers = max_workers
        self._print_lock = Lock()

    def _process_audio_file(self, filepath: str) -> Tuple[str, float, int, str]:
        duration = self.get_audio_duration(filepath=filepath)
        size = self.get_file_size(filepath=filepath)
        ext = Path(filepath).suffix.lower()

        return (filepath, duration, size, ext)

    @staticmethod
    def get_audio_duration(filepath: str) -> float:
        try:
            info = sf.info(file=filepath)
            return info.duration
        except Exception as e:
            logger.error(f"Warning: Could not get duration for {filepath}: {e}")
            return 0.0

    @staticmethod
    def get_file_size(filepath: str) -> int:
        try:
            return os.path.getsize(filename=filepath)
        except Exception as e:
            logger.error(f"Warning: Could not get filesize for {filepath}: {e}")
            return 0

    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds <= 0 or seconds != seconds:  # Check for 0, negative, or NaN
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return "".join(parts)

    @staticmethod
    def format_size(bytes: int) -> str:
        if bytes <= 0:
            return "0.00 B"

        size = float(bytes)

        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0

        return f"{size:.2f} PB"

    def generate_summary(
        self,
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
        save_json: bool = False,
        json_output: str = "audio_summary.json",
    ) -> Dict:
        """
        Generate comprehensive summary statistics for audio directory.

        Args:
            root_dir: Root directory containing voice subdirectories
            extensions: Tuple of audio file extensions to process
            save_json: Whether to save summary as JSON file
            json_output: Path for JSON output file

        Returns:
            Dictionary with summary statistics
        """
        root_path = Path(self.root_dir)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {self.root_dir}")

        print(f"Analyzing audio files in: {self.root_dir}")
        print(f"Using ThreadPoolExecutor with max_workers={self.max_workers or 'auto'}")
        print("=" * 70)

        # Initialize counters
        total_duration = 0.0
        total_size = 0
        total_files = 0
        total_folders = 0

        folder_stats = {}
        extension_stats = defaultdict(lambda: {"count": 0, "duration": 0.0, "size": 0})
        durations = []
        sizes = []

        # Iterate through subdirectories
        subdirs = [d for d in root_path.iterdir() if d.is_dir()]
        total_folders = len(subdirs)

        logger.info(f"Found {total_folders} folders. Processing...\n")

        all_files_info = []
        for subdir in sorted(subdirs):
            voice_idx = subdir.name
            audio_files = []
            for ext in extensions:
                audio_files.extend(subdir.glob(f"*{ext}"))
            for audio_file in audio_files:
                all_files_info.append((str(audio_file), voice_idx))

        file_paths = [info[0] for info in all_files_info]
        filepath_to_folder = {info[0]: info[1] for info in all_files_info}

        results = []
        completed = 0

        total_files = len(file_paths)
        logger.info(f"Found {total_files} total files to process.\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_audio_file, path): path for path in file_paths
            }
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if completed % 100 == 0 or completed == total_files:
                        logger.info(
                            f"  Processed {completed}/{total_files} files ({completed*100//total_files}%)"
                        )
                except Exception as e:
                    path = future_to_path[future]
                    logger.error(f"Error processing {path}: {e}")

        # initialize folder stats
        for subdir in sorted(subdirs):
            folder_stats[subdir.name] = {
                "file_count": 0,
                "total_duration": 0.0,
                "total_size": 0,
                "avg_duration": 0.0,
                "avg_size": 0,
            }

        for filepath, duration, size, ext in results:
            folder_name = filepath_to_folder[filepath]

            total_duration += duration
            total_size += size

            extension_stats[ext]["count"] += 1
            extension_stats[ext]["duration"] += duration
            extension_stats[ext]["size"] += size

            durations.append(duration)
            sizes.append(size)

            folder_stats[folder_name]["file_count"] += 1
            folder_stats[folder_name]["total_duration"] += duration
            folder_stats[folder_name]["total_size"] += size

        for folder_name, stats in folder_stats.items():
            if stats["file_count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["file_count"]
                stats["avg_size"] = stats["total_size"] / stats["file_count"]

        # Print folder summaries
        logger.info("Folder Summaries:")
        for i, (folder_name, stats) in enumerate(sorted(folder_stats.items()), 1):
            logger.info(
                f"[{i}/{total_folders}] {folder_name}: {stats['file_count']} files, "
                f"{self.format_duration(stats['total_duration'])}, {self.format_size(stats['total_size'])}"
            )

        # Calculate statistics
        avg_duration = total_duration / total_files if total_files > 0 else 0
        avg_size = total_size / total_files if total_files > 0 else 0

        # Find min/max durations
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        # Find folders with most/least files
        folders_by_count = sorted(
            folder_stats.items(), key=lambda x: x[1]["file_count"], reverse=True
        )
        most_files_folder = folders_by_count[0] if folders_by_count else None
        least_files_folder = folders_by_count[-1] if folders_by_count else None

        # Find folders with most/least duration
        folders_by_duration = sorted(
            folder_stats.items(), key=lambda x: x[1]["total_duration"], reverse=True
        )
        most_duration_folder = folders_by_duration[0] if folders_by_duration else None

        # Create summary dictionary
        summary = {
            "root_directory": str(self.root_dir),
            "total_folders": total_folders,
            "total_files": total_files,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_formatted": self.format_duration(total_duration),
            "total_size_bytes": total_size,
            "total_size_formatted": self.format_size(total_size),
            "average_duration_seconds": round(avg_duration, 2),
            "average_duration_formatted": self.format_duration(avg_duration),
            "average_size_bytes": int(avg_size),
            "average_size_formatted": self.format_size(avg_size),
            "min_duration_seconds": round(min_duration, 2),
            "max_duration_seconds": round(max_duration, 2),
            "files_per_folder": round(total_files / total_folders, 2) if total_folders > 0 else 0,
            "extension_breakdown": {
                ext: {
                    "count": stats["count"],
                    "duration_seconds": round(stats["duration"], 2),
                    "duration_formatted": self.format_duration(stats["duration"]),
                    "size_bytes": stats["size"],
                    "size_formatted": self.format_size(stats["size"]),
                }
                for ext, stats in extension_stats.items()
            },
            "folder_statistics": {
                folder: {
                    "file_count": stats["file_count"],
                    "total_duration_seconds": round(stats["total_duration"], 2),
                    "total_duration_formatted": self.format_duration(stats["total_duration"]),
                    "total_size_bytes": stats["total_size"],
                    "total_size_formatted": self.format_size(stats["total_size"]),
                    "avg_duration_seconds": round(stats["avg_duration"], 2),
                    "avg_size_bytes": int(stats["avg_size"]),
                }
                for folder, stats in folder_stats.items()
            },
        }

        # Add insights
        if most_files_folder:
            summary["insights"] = {
                "folder_with_most_files": {
                    "folder": most_files_folder[0],
                    "count": most_files_folder[1]["file_count"],
                },
                "folder_with_least_files": (
                    {"folder": least_files_folder[0], "count": least_files_folder[1]["file_count"]}
                    if least_files_folder
                    else None
                ),
                "folder_with_most_duration": (
                    {
                        "folder": most_duration_folder[0],
                        "duration_seconds": round(most_duration_folder[1]["total_duration"], 2),
                        "duration_formatted": self.format_duration(
                            most_duration_folder[1]["total_duration"]
                        ),
                    }
                    if most_duration_folder
                    else None
                ),
            }

        # Print summary
        self.print_summary(summary)

        # Save to JSON if requested
        if self.save_json:
            with open(self.json_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Summary saved to: {self.json_output}")

        return summary

    def print_summary(self, summary: Dict):
        """
        Print formatted summary to console.

        Args:
            summary: Summary dictionary
        """
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        print(f"\nDirectory Structure:")
        print(f"  Total Folders: {summary['total_folders']}")
        print(f"  Total Audio Files: {summary['total_files']}")
        print(f"  Average Files per Folder: {summary['files_per_folder']:.2f}")

        print(f"\nDuration:")
        print(
            f"  Total Duration: {summary['total_duration_formatted']} ({summary['total_duration_seconds']:.2f}s)"
        )
        print(
            f"  Average Duration per File: {summary['average_duration_formatted']} ({summary['average_duration_seconds']:.2f}s)"
        )
        print(f"  Shortest File: {summary['min_duration_seconds']:.2f}s")
        print(f"  Longest File: {summary['max_duration_seconds']:.2f}s")

        print(f"\nStorage:")
        print(
            f"  Total Size: {summary['total_size_formatted']} ({summary['total_size_bytes']:,} bytes)"
        )
        print(
            f"  Average Size per File: {summary['average_size_formatted']} ({summary['average_size_bytes']:,} bytes)"
        )

        if summary.get("extension_breakdown"):
            print(f"\nFile Type Breakdown:")
            for ext, stats in summary["extension_breakdown"].items():
                print(f"  {ext}:")
                print(f"    Count: {stats['count']} files")
                print(
                    f"    Duration: {stats['duration_formatted']} ({stats['duration_seconds']:.2f}s)"
                )
                print(f"    Size: {stats['size_formatted']} ({stats['size_bytes']:,} bytes)")

        if summary.get("insights"):
            print(f"\nInsights:")
            insights = summary["insights"]
            if insights.get("folder_with_most_files"):
                print(
                    f"  Folder with most files: {insights['folder_with_most_files']['folder']} ({insights['folder_with_most_files']['count']} files)"
                )
            if insights.get("folder_with_least_files"):
                print(
                    f"  Folder with least files: {insights['folder_with_least_files']['folder']} ({insights['folder_with_least_files']['count']} files)"
                )
            if insights.get("folder_with_most_duration"):
                print(
                    f"  Folder with most audio: {insights['folder_with_most_duration']['folder']} ({insights['folder_with_most_duration']['duration_formatted']})"
                )

        print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # Initialize summarizer
    summarizer = AudioDirectorySummary(
        root_dir="data/kmong",
        extensions=(".wav", ".mp3", ".flac"),
        save_json=True,
        json_output="audio_summary.json",
    )

    # Generate summary
    summary = summarizer.generate_summary()

    # Access specific statistics
    print(f"\nQuick Stats:")
    print(f"Total audio time: {summary['total_duration_formatted']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total folders: {summary['total_folders']}")
    print("\n")
