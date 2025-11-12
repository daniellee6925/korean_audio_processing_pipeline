import os
import json
import librosa
from pathlib import Path
from typing import Optional, Dict, List


class AudioMetadataUpdater:
    """
    Updates existing JSON metadata by adding source_name, filepath, and audio_length.
    Expected structure: rootfolder/voice_idx/
    - ONE JSON file per folder named foldername.json (e.g., voice_001/voice_001.json)
    - JSON contains: voice_idx, url, metadata
    - Multiple audio files in the same folder
    Creates one metadata entry per audio file, all sharing the same base metadata.
    """

    def __init__(self):
        """Initialize the metadata updater."""
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
            return round(duration, 2)
        except Exception as e:
            print(f"Warning: Could not get duration for {filepath}: {e}")
            return 0.0

    def find_audio_files(
        self, folder_path: Path, extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    ) -> List[Path]:
        """
        Find all audio files in a folder.

        Args:
            folder_path: Path to folder
            extensions: Tuple of audio file extensions to find

        Returns:
            List of audio file paths
        """
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder_path.glob(f"*{ext}"))

        return sorted(audio_files)

    def update_json_metadata(
        self,
        json_filepath: str,
        source_name: str,
        folder_path: Path,
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    ) -> List[Dict]:
        """
        Update a single JSON file by expanding it into multiple entries,
        one for each audio file in the folder.

        Args:
            json_filepath: Path to JSON file
            source_name: Name of the root folder (source)
            folder_path: Path to the folder containing audio files
            extensions: Tuple of audio file extensions to find

        Returns:
            List of metadata dictionaries, one per audio file
        """
        json_path = Path(json_filepath)
        all_entries = []

        try:
            # Load existing JSON
            with open(json_path, "r", encoding="utf-8") as f:
                base_metadata = json.load(f)

            # Find all audio files in the folder
            audio_files = self.find_audio_files(folder_path, extensions)

            if not audio_files:
                print(f"  Warning: No audio files found in {folder_path}")
                return []

            print(f"  Found {len(audio_files)} audio files")

            # Create one metadata entry per audio file
            for audio_file in audio_files:
                # Copy base metadata
                metadata = base_metadata.copy()

                # Add missing fields
                metadata["source_name"] = source_name
                metadata["filepath"] = str(audio_file)
                metadata["audio_length"] = self.get_audio_duration(str(audio_file))

                all_entries.append(metadata)

            return all_entries

        except json.JSONDecodeError as e:
            print(f"  Error: Invalid JSON in {json_filepath}: {e}")
            return []
        except Exception as e:
            print(f"  Error: Could not process {json_filepath}: {e}")
            return []

    def process_directory(
        self,
        root_dir: str,
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
        create_individual_jsons: bool = False,
    ) -> List[Dict]:
        """
        Process all folders in root directory. Each folder has ONE JSON file
        named after the folder (e.g., voice_001/voice_001.json).
        This JSON will be expanded into multiple entries (one per audio file).

        Expected structure:
        root_dir/
            voice_idx_1/
                voice_idx_1.json  (has voice_idx, url, metadata)
                audio_file_1.wav
                audio_file_2.wav
            voice_idx_2/
                voice_idx_2.json
                audio_file_1.wav

        Args:
            root_dir: Root directory containing voice subdirectories
            extensions: Tuple of audio file extensions to process
            create_individual_jsons: If True, create individual JSON files for each audio

        Returns:
            List of all metadata dictionaries (one per audio file)
        """
        root_path = Path(root_dir)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        # Get source name from root folder
        source_name = root_path.name

        all_metadata = []
        processed_folders = 0
        processed_files = 0
        error_count = 0

        print(f"Processing folders in: {root_dir}")
        print(f"Source name: {source_name}")
        print(f"Looking for JSON files named after their folder")
        print("=" * 60)

        # Iterate through subdirectories (voice indices)
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue

            voice_idx = subdir.name
            # JSON file is named after the folder
            json_path = subdir / f"{voice_idx}.json"

            if not json_path.exists():
                print(f"\nSkipping {voice_idx}: No {voice_idx}.json found")
                continue

            print(f"\nProcessing voice_idx: {voice_idx}")

            try:
                # Get metadata entries for all audio files in this folder
                metadata_entries = self.update_json_metadata(
                    json_filepath=str(json_path),
                    source_name=source_name,
                    folder_path=subdir,
                    extensions=extensions,
                )

                if metadata_entries:
                    all_metadata.extend(metadata_entries)
                    processed_folders += 1
                    processed_files += len(metadata_entries)
                    print(f"  ✓ Created {len(metadata_entries)} metadata entries")

                    # Optionally create individual JSON files for each audio
                    if create_individual_jsons:
                        for entry in metadata_entries:
                            audio_path = Path(entry["filepath"])
                            json_out_path = audio_path.with_suffix(".json")

                            with open(json_out_path, "w", encoding="utf-8") as f:
                                json.dump(entry, f, indent=2, ensure_ascii=False)

                            print(f"    → {json_out_path.name}")
                else:
                    error_count += 1

            except Exception as e:
                error_count += 1
                print(f"  ✗ Error processing {voice_idx}: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Folders processed: {processed_folders}")
        print(f"Total audio files processed: {processed_files}")
        print(f"Errors: {error_count}")

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
        all_metadata = self.process_directory(root_dir, extensions=extensions)

        # Write master JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Master metadata file created: {output_path}")
        print(f"  Total entries: {len(all_metadata)}")


if __name__ == "__main__":
    # Initialize updater
    updater = AudioMetadataUpdater()

    # Process directory - reads foldername.json from each folder
    # Creates entries for all audio files in that folder
    metadata_list = updater.process_directory(
        root_dir="data/kmong",
        extensions=(".wav",),
        create_individual_jsons=True,  # Set to True to create individual JSONs per audio
    )

    # Or create a single master JSON file with all metadata
    # updater.generate_master_json(
    #     root_dir="voice_casting",
    #     output_path="all_audio_metadata.json"
    # )
