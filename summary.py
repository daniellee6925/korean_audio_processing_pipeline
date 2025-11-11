import os
import librosa
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json


class AudioDirectorySummary:
    """
    Generates summary statistics for audio files in a directory structure.
    """

    def __init__(self):
        """Initialize the summary calculator."""
        pass

    def get_audio_duration(self, filepath: str) -> float:
        """
        Get audio duration in seconds.

        Args:
            filepath: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=filepath)
            return duration
        except Exception as e:
            print(f"Warning: Could not get duration for {filepath}: {e}")
            return 0.0

    def get_file_size(self, filepath: str) -> int:
        """
        Get file size in bytes.

        Args:
            filepath: Path to file

        Returns:
            File size in bytes
        """
        try:
            return os.path.getsize(filepath)
        except Exception:
            return 0

    def format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "2h 30m 45s")
        """
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

        return " ".join(parts)

    def format_size(self, bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            bytes: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 GB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"

    def generate_summary(
        self,
        root_dir: str,
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
        root_path = Path(root_dir)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        print(f"Analyzing audio files in: {root_dir}")
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

        print(f"Found {total_folders} folders. Processing...\n")

        for i, subdir in enumerate(sorted(subdirs), 1):
            voice_idx = subdir.name
            print(f"[{i}/{total_folders}] Processing folder: {voice_idx}")

            # Find all audio files in this subdirectory
            audio_files = []
            for ext in extensions:
                audio_files.extend(subdir.glob(f"*{ext}"))

            folder_duration = 0.0
            folder_size = 0
            folder_file_count = len(audio_files)

            # Process each audio file
            for audio_file in audio_files:
                duration = self.get_audio_duration(str(audio_file))
                size = self.get_file_size(str(audio_file))
                ext = audio_file.suffix.lower()

                # Update totals
                total_duration += duration
                total_size += size
                folder_duration += duration
                folder_size += size

                # Track by extension
                extension_stats[ext]["count"] += 1
                extension_stats[ext]["duration"] += duration
                extension_stats[ext]["size"] += size

                # Track individual durations and sizes
                durations.append(duration)
                sizes.append(size)

            total_files += folder_file_count

            # Store folder statistics
            folder_stats[voice_idx] = {
                "file_count": folder_file_count,
                "total_duration": folder_duration,
                "total_size": folder_size,
                "avg_duration": folder_duration / folder_file_count if folder_file_count > 0 else 0,
                "avg_size": folder_size / folder_file_count if folder_file_count > 0 else 0,
            }

            print(
                f"  Files: {folder_file_count}, Duration: {self.format_duration(folder_duration)}, Size: {self.format_size(folder_size)}"
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
            "root_directory": str(root_dir),
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
        if save_json:
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Summary saved to: {json_output}")

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

        print(f"\n⏱Duration:")
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
    summarizer = AudioDirectorySummary()

    # Generate summary
    summary = summarizer.generate_summary(
        root_dir="voice_pick",
        extensions=(".wav", ".mp3", ".flac"),
        save_json=True,
        json_output="audio_summary.json",
    )

    # Access specific statistics
    print(f"\nQuick Stats:")
    print(f"Total audio time: {summary['total_duration_formatted']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total folders: {summary['total_folders']}")
    print("\n")
