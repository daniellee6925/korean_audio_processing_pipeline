import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List
import os
import shutil
from pathlib import Path


class FilterByBGM:
    """
    Detects background music in audio files by analyzing silence patterns.
    Speech-only audio has clear silence gaps, while music creates continuous sound.
    """

    def __init__(
        self,
        root_dir: str,
        bgm_dir: str = "bgm found",
        frame_length: int = 2048,
        hop_length: int = 512,
        energy_threshold: float = 0.02,
        min_silence_duration: float = 0.3,
        silence_threshold: int = 3,
        total_silence_threshold: float = 1.5,
        check_duration: float = 20.0,
        extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
        recursive: bool = True,
        move_files: bool = True,
    ):
        """
        Initialize the detector.

        Args:
            frame_length: Number of samples per frame
            hop_length: Number of samples between frames
            energy_threshold: Threshold for silence detection (lower = more sensitive)
            min_silence_duration: Minimum duration (seconds) to count as silence
        """
        self.root_dir = root_dir
        self.bgm_dir = bgm_dir
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_silence_duration = min_silence_duration
        self.check_duration = check_duration
        self.extensions = extensions
        self.recursive = recursive
        self.move_files = move_files
        self.silence_threshold = silence_threshold
        self.total_silence_threshold = total_silence_threshold

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return first N seconds."""
        audio, sr = librosa.load(filepath, sr=None, duration=self.check_duration)
        return audio, sr

    def compute_energy(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute frame-wise energy (RMS) of the audio."""
        rms = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]

        # Normalize energy
        if np.max(rms) > 0:
            rms = rms / np.max(rms)

        return rms

    def detect_silence_segments(self, energy: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect silence segments based on energy threshold.

        Returns:
            List of (start_time, end_time) tuples for silence segments
        """
        times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=self.hop_length)

        is_silent = energy < self.energy_threshold

        segments = []
        start_idx = None

        for i, silent in enumerate(is_silent):
            if silent and start_idx is None:
                start_idx = i
            elif not silent and start_idx is not None:
                duration = times[i] - times[start_idx]
                if duration >= self.min_silence_duration:
                    segments.append((times[start_idx], times[i]))
                start_idx = None

        # Handle case where silence extends to end
        if start_idx is not None:
            duration = times[-1] - times[start_idx]
            if duration >= self.min_silence_duration:
                segments.append((times[start_idx], times[-1]))

        return segments

    def move_to_bgm_folder(
        self,
        filepath: str,
    ) -> str:
        """
        Move file to bgm_music folder if it has background music.

        Args:
            filepath: Original file path
            bgm_folder: Name of the folder to move files with BGM

        Returns:
            New file path after moving
        """
        # Get the directory of the original file
        filename = os.path.basename(filepath)

        # Create bgm_music folder in the same directory as the original file
        os.makedirs(self.bgm_dir, exist_ok=True)

        # Define destination path
        dest_path = os.path.join(self.bgm_dir, filename)

        # Handle duplicate filenames
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(self.bgm_dir, f"{base}_{counter}{ext}")
                counter += 1

        # Move the file
        shutil.move(filepath, dest_path)
        return dest_path

    def classify(
        self,
        filepath: str,
    ) -> dict:
        """
        Classify if audio has background music and optionally move file.

        Args:
            filepath: Path to audio file

        Returns:
            Dictionary with classification results and metrics
        """
        # Load audio
        audio, sr = self.load_audio(filepath)

        # Compute energy
        energy = self.compute_energy(audio, sr)

        # Detect silence segments
        silence_segments = self.detect_silence_segments(energy, sr)

        # Calculate metrics
        num_silences = len(silence_segments)
        total_silence = sum(end - start for start, end in silence_segments)

        # Classification logic
        has_background_music = (
            num_silences < self.silence_threshold or total_silence < self.total_silence_threshold
        )

        result = {
            "has_background_music": has_background_music,
            "classification": "Music + Speech" if has_background_music else "Speech Only",
            "num_silence_segments": num_silences,
            "total_silence_duration": round(total_silence, 2),
            "silence_segments": [(round(s, 2), round(e, 2)) for s, e in silence_segments],
            "audio_duration": round(len(audio) / sr, 2),
            "original_path": filepath,
        }

        # Move file if it has background music
        if has_background_music and self.move_files:
            new_path = self.move_to_bgm_folder(filepath)
            result["moved_to"] = new_path
            result["file_moved"] = True
        else:
            result["moved_to"] = filepath
            result["file_moved"] = False

        return result

    def analyze_batch(
        self,
        filepaths: List[str],
    ) -> List[dict]:
        """
        Analyze multiple audio files and optionally move those with BGM.

        Args:
            filepaths: List of audio file paths
            bgm_folder: Name of folder for BGM files
        """
        results = []
        for filepath in filepaths:
            try:
                result = self.classify(filepath)
                result["filepath"] = filepath
                results.append(result)
            except Exception as e:
                results.append({"filepath": filepath, "error": str(e)})
        return results

    def process_all(
        self,
    ) -> dict:
        """
        Process all audio files in a directory.
        Returns:
            Dictionary with summary statistics and detailed results
        """
        root_path = Path(self.root_dir)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {self.root_dir}")

        # Find all audio files
        audio_files = []
        if self.recursive:
            for ext in self.extensions:
                audio_files.extend(root_path.rglob(f"*{ext}"))
        else:
            for ext in self.extensions:
                audio_files.extend(root_path.glob(f"*{ext}"))

        audio_files = [str(f) for f in audio_files]

        if not audio_files:
            return {
                "total_files": 0,
                "files_processed": 0,
                "files_with_bgm": 0,
                "files_without_bgm": 0,
                "errors": 0,
                "results": [],
            }

        print(f"Found {len(audio_files)} audio files in {self.root_dir}")
        print("Processing...")

        # Process all files
        results = []
        files_with_bgm = 0
        files_without_bgm = 0
        errors = 0

        for i, filepath in enumerate(audio_files, 1):
            try:
                print(f"[{i}/{len(audio_files)}] Processing: {os.path.basename(filepath)}")
                result = self.classify(filepath)
                result["filepath"] = filepath
                results.append(result)

                if result["has_background_music"]:
                    files_with_bgm += 1
                    if result["file_moved"]:
                        print(f"  ✓ BGM detected - Moved to: {result['moved_to']}")
                else:
                    files_without_bgm += 1
                    print(f"  ✓ Speech only - File kept in place")

            except Exception as e:
                errors += 1
                print(f"  ✗ Error: {str(e)}")
                results.append({"filepath": filepath, "error": str(e)})

        summary = {
            "total_files": len(audio_files),
            "files_processed": len(audio_files) - errors,
            "files_with_bgm": files_with_bgm,
            "files_without_bgm": files_without_bgm,
            "errors": errors,
            "results": results,
        }

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total files found: {summary['total_files']}")
        print(f"Successfully processed: {summary['files_processed']}")
        print(f"Files with BGM (moved): {summary['files_with_bgm']}")
        print(f"Files without BGM (kept): {summary['files_without_bgm']}")
        print(f"Errors: {summary['errors']}")

        return summary


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FilterByBGM(
        root_dir="audio_files_3 copy",
        energy_threshold=0.02,  # Adjust based on your audio
        min_silence_duration=0.3,  # Minimum 0.3s silence
        silence_threshold=2,
    )

    summary = detector.process_all()
