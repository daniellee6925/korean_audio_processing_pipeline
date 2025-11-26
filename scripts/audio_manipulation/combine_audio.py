import subprocess
import re
from pathlib import Path
from typing import Union


class AudioCombiner:
    """
    A class for concatenating audio files using ffmpeg.
    Automatically sorts files by numeric suffix (e.g., name_1, name_2, name_3).
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize the AudioCombiner.

        Args:
            ffmpeg_path: Path to ffmpeg executable (default: "ffmpeg" assumes it's in PATH)
        """
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if ffmpeg is available."""
        try:
            subprocess.run([self.ffmpeg_path, "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"ffmpeg not found at '{self.ffmpeg_path}'. "
                "Please install ffmpeg or provide the correct path."
            )

    def _extract_number(self, filename: str) -> int:
        """
        Extract numeric suffix from filename.

        Args:
            filename: The filename to extract number from

        Returns:
            The extracted number, or 0 if no number found
        """
        # Match pattern like "name_123" or "file_5"
        match = re.search(r"_(\d+)", filename)
        if match:
            return int(match.group(1))
        return 0

    def concatenate_from_directory(
        self,
        input_dir: Union[str, Path],
        output_file: Union[str, Path],
        pattern: str = "*",
        audio_codec: str = "copy",
        overwrite: bool = True,
    ) -> bool:
        """
        Concatenate all audio files from a directory in numeric order.
        Files are sorted by numeric suffix (e.g., name_1, name_2, name_3).

        Args:
            input_dir: Directory containing audio files
            output_file: Output file path
            pattern: Glob pattern to match files (default: "*" matches all files)
            audio_codec: Audio codec to use (default: "copy" for no re-encoding)
            overwrite: Whether to overwrite existing output file

        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        if not input_path.is_dir():
            raise ValueError(f"Path is not a directory: {input_dir}")

        # Get all files matching the pattern
        files = list(input_path.glob(pattern))

        # Filter out non-files and directories
        files = [f for f in files if f.is_file()]

        if not files:
            raise ValueError(f"No files found in directory: {input_dir}")

        # Sort files by numeric suffix
        files.sort(key=lambda x: self._extract_number(x.name))

        print(f"Found {len(files)} files to concatenate:")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f.name}")

        # Create a temporary file list for ffmpeg concat demuxer
        output_path = Path(output_file)
        list_file = output_path.parent / "concat_list.txt"

        try:
            # Write file list
            with open(list_file, "w", encoding="utf-8") as f:
                for file in files:
                    # Use absolute paths to avoid issues
                    abs_path = file.resolve()
                    # Escape single quotes in file paths
                    escaped_path = str(abs_path).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Build ffmpeg command
            cmd = [
                self.ffmpeg_path,
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:a",
                audio_codec,
            ]

            if overwrite:
                cmd.append("-y")

            cmd.append(str(output_file))

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return False

            print(f"\nSuccessfully concatenated {len(files)} files to {output_file}")
            return True

        finally:
            # Clean up temporary list file
            if list_file.exists():
                list_file.unlink()


if __name__ == "__main__":
    # Example usage
    combiner = AudioCombiner()

    # Concatenate all audio files from a directory
    # Files will be sorted by numeric suffix: name_1, name_2, name_3, etc.
    combiner.concatenate_from_directory(input_dir="segs/seg3", output_file="combined_output3.wav")

    # You can also specify a pattern to match specific files
    # combiner.concatenate_from_directory(
    #     input_dir="path/to/audio/files",
    #     output_file="combined_output.mp3",
    #     pattern="audio_*.mp3"  # Only match files like "audio_1.mp3", "audio_2.mp3"
    # )

    print("AudioCombiner class ready to use!")
