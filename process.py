import os
import json
import librosa
import csv
from pathlib import Path
from typing import Optional, Dict, List
import re


class AudioMetadataGenerator:
    """
    Generates JSON metadata files for audio files in a directory structure.
    Expected structure: rootfolder/voice_idx/audio_files
    Each voice_idx folder contains a data_labels.csv file with labels.
    """

    def __init__(self, base_url: str = "https://example.com"):
        """
        Initialize the metadata generator.

        Args:
            base_url: Base URL to be used for generating audio URLs
        """
        self.base_url = base_url.rstrip("/")

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
            return round(duration, 2)
        except Exception as e:
            print(f"Warning: Could not get duration for {filepath}: {e}")
            return 0.0

    def load_data_labels(self, csv_path: str) -> Optional[str]:
        """
        Load data labels CSV file and return the label from the first data row.

        Expected CSV format:
        voice_actor_idx, labels
        6, 지적인, 무게있는, 차분한, 내레이션

        Args:
            csv_path: Path to data_labels.csv

        Returns:
            Label string or None if not found
        """
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)

                # Skip header row
                header = next(reader, None)

                # Read first data row
                for row in reader:
                    # Skip empty rows
                    if not row or len(row) < 2:
                        continue

                    # Skip rows that are all empty
                    if all(not cell.strip() for cell in row):
                        continue

                    # Column 0: voice_actor_idx (ignore)
                    # Column 1: labels
                    labels = str(row[1]).strip()

                    if labels:
                        print(f"  Loaded label: {labels}")
                        return labels

            print(f"  Warning: No valid label found in CSV")
            return None

        except FileNotFoundError:
            print(f"  Warning: CSV file not found: {csv_path}")
            return None
        except Exception as e:
            print(f"  Warning: Could not load {csv_path}: {e}")
            return None

    def generate_metadata(
        self, audio_filepath: str, source_name: str, voice_idx: str, label: Optional[str]
    ) -> Dict:
        """
        Generate metadata dictionary for a single audio file.

        Args:
            audio_filepath: Full path to audio file
            source_name: Name of the root folder (source)
            voice_idx: Voice index (subfolder name)
            label: Label string for this voice_idx

        Returns:
            Dictionary with metadata
        """
        # Generate URL
        url = f"{self.base_url}/{voice_idx}"

        # Get audio duration
        audio_length = self.get_audio_duration(audio_filepath)

        return {
            "source_name": source_name,
            "filepath": audio_filepath,
            "voice_idx": voice_idx,
            "url": url,
            "metadata": label,
            "audio_length": audio_length,
        }

    def process_directory(
        self,
        root_dir: str,
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Process all audio files in root directory and generate metadata JSON files.

        Expected structure:
        root_dir/
            voice_idx_1/
                data_labels.csv  (with voice_actor_idx, labels columns)
                audio_file_1.wav
                audio_file_2.wav
            voice_idx_2/
                data_labels.csv
                audio_file_1.wav

        Args:
            root_dir: Root directory containing voice subdirectories
            extensions: Tuple of audio file extensions to process
            output_dir: Optional directory to save JSON files (defaults to same as audio)

        Returns:
            List of all generated metadata dictionaries
        """
        root_path = Path(root_dir)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        # Get source name from root folder
        source_name = root_path.name

        all_metadata = []
        processed_count = 0
        error_count = 0

        print(f"Processing audio files in: {root_dir}")
        print(f"Source name: {source_name}")
        print("=" * 60)

        # Iterate through subdirectories (voice indices)
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue

            voice_idx = subdir.name
            print(f"\nProcessing voice_idx: {voice_idx}")

            # Load label from data_labels.csv in this folder
            csv_path = subdir / "data_labels.csv"
            label = self.load_data_labels(str(csv_path))

            if not label:
                print(f"  Warning: No label found in {csv_path}")

            # Find all audio files in this subdirectory
            audio_files = []
            for ext in extensions:
                audio_files.extend(subdir.glob(f"*{ext}"))

            print(f"  Found {len(audio_files)} audio files")

            # Process each audio file (all get the same label)
            for audio_file in sorted(audio_files):
                try:
                    # Generate metadata
                    metadata = self.generate_metadata(
                        audio_filepath=str(audio_file),
                        source_name=source_name,
                        voice_idx=voice_idx,
                        label=label,
                    )

                    all_metadata.append(metadata)

                    # Determine output path for JSON
                    if output_dir:
                        output_path = Path(output_dir) / voice_idx
                        output_path.mkdir(parents=True, exist_ok=True)
                        json_filename = audio_file.stem + ".json"
                        json_path = output_path / json_filename
                    else:
                        json_path = audio_file.with_suffix(".json")

                    # Write JSON file
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    processed_count += 1
                    print(f"  ✓ {audio_file.name} -> {json_path.name}")

                except Exception as e:
                    error_count += 1
                    print(f"  ✗ Error processing {audio_file.name}: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total files processed: {processed_count}")
        print(f"Errors: {error_count}")
        print(f"Metadata files created: {processed_count}")

        return all_metadata

    def generate_master_json(
        self,
        root_dir: str,
        output_path: str = "master_metadata.json",
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    ):
        """
        Generate a single master JSON file containing all metadata.

        Args:
            root_dir: Root directory containing voice subdirectories
            output_path: Path for the master JSON file
            extensions: Tuple of audio file extensions to process
        """
        all_metadata = self.process_directory(root_dir, extensions, output_dir=None)

        # Write master JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Master metadata file created: {output_path}")
        print(f"  Total entries: {len(all_metadata)}")


if __name__ == "__main__":
    # Initialize generator
    generator = AudioMetadataGenerator(base_url="https://vcast.audien.com/artist/")

    # Process directory and create individual JSON files for each audio
    metadata_list = generator.process_directory(root_dir="voice_casting", extensions=(".wav",))

    # Or create a single master JSON file with all metadata
    # generator.generate_master_json(
    #     root_dir="/path/to/rootfolder",
    #     output_path="all_audio_metadata.json"
    # )
