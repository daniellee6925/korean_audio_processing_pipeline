import os
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import concurrent.futures
from loguru import logger


class AudioDirectorySummary:
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
    def format_file_size(bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"

    def process_file(self, filepath: str) -> Tuple[str, float, int]:
        duration = self.get_audio_duration(filepath)
        size = self.get_file_size(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        return ext, duration, size

    def generate_summary(
        self,
    ):
        root_path = Path(self.root_dir)
        if not root_path.exists():
            raise ValueError(f"Directory deos not exist: {self.root_dir}")

        logger.info(f"Analyzing audio files in {self.root_dir}")

        # initialize counters
        total_duration = 0.0
        total_size = 0
        total_files = 0

        extension_stats = defaultdict(lambda: {"count": 0, "duration": 0.0, "size": 0})

        files = [
            f for f in root_path.rglob("*") if f.is_file() and f.suffix.lower() in self.extensions
        ]

        total_files = len(files)
        logger.info(f"Found {total_files} audio files under {self.root_dir}")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for res in executor.map(self.process_file, map(str, files)):
                results.append(res)

        for ext, duration, size in results:
            extension_stats[ext]["count"] += 1
            extension_stats[ext]["count"] += 1
            extension_stats[ext]["duration"] += duration
            extension_stats[ext]["size"] += size
            total_duration += duration
            total_size += size

        avg_duration = total_duration / total_files if total_files else 0
        avg_size = total_size / total_files if total_files else 0

        summary = {
            "root_directory": str(root_path),
            "total_files": total_files,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_formatted": self.format_duration(total_duration),
            "total_size_bytes": total_size,
            "total_size_formatted": self.format_file_size(total_size),
            "average_duration_seconds": round(avg_duration, 2),
            "average_duration_formatted": self.format_duration(avg_duration),
            "average_size_bytes": avg_size,
            "average_size_formatted": self.format_file_size(avg_size),
            "extension_breakdown": {
                ext: {
                    "count": stats["count"],
                    "duration_seconds": round(stats["duration"], 2),
                    "duration_formatted": self.format_duration(stats["duration"]),
                    "size_bytes": stats["size"],
                    "size_formatted": self.format_file_size(stats["size"]),
                }
                for ext, stats in extension_stats.items()
            },
        }

        if self.save_json:
            with open(self.json_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved summmary to {self.json_output}")

        self.print_summary(summary)

        return summary

    def print_summary(self, summary: dict) -> None:
        """Print formatted audio summary to console."""

        print("\n" + "=" * 70)
        print("AUDIO DIRECTORY SUMMARY")
        print("=" * 70)

        print(f"\nRoot Directory: {summary['root_directory']}")
        print(f"Total Files: {summary['total_files']:,}")
        print(
            f"Total Duration: {summary['total_duration_formatted']} ({summary['total_duration_seconds']:.2f} sec)"
        )
        print(
            f"Total Size: {summary['total_size_formatted']} ({summary['total_size_bytes']:,} bytes)"
        )
        print(
            f"Average Duration: {summary['average_duration_formatted']} ({summary['average_duration_seconds']:.2f} sec)"
        )
        print(
            f"Average File Size: {summary['average_size_formatted']} ({summary['average_size_bytes']:,} bytes)"
        )

        print("\nFile Type Breakdown:")
        print("-" * 70)
        for ext, stats in summary["extension_breakdown"].items():
            print(f"Extension: {ext}")
            print(f"  Count: {stats['count']:,}")
            print(
                f"  Duration: {stats['duration_formatted']} ({stats['duration_seconds']:.2f} sec)"
            )
            print(f"  Size: {stats['size_formatted']} ({stats['size_bytes']:,} bytes)")
            print("-" * 70)

        print("\nSUMMARY:")
        print(f"  Total Duration: {summary['total_duration_formatted']}")
        print(f"  Total Files: {summary['total_files']:,}")
        print(f"  Total Size: {summary['total_size_formatted']}")
        print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Initialize summarizer
    summarizer = AudioDirectorySummary(root_dir="Recording_251111_1_Segments")

    # Generate summary
    summary = summarizer.generate_summary()

    # Access specific statistics
    print(f"\nQuick Stats:")
    print(f"Total audio time: {summary['total_duration_formatted']}")
    print(f"Total files: {summary['total_files']}")
    print("\n")
