"""
sentence_16/cleaned/
├── 01/
│   ├── segment_1.wav
│   ├── segment_1.csv
│   └── ...
├── 02/
│   ├── segment_2.wav
│   ├── segment_2.csv
│   └── ...
└── a/
├── file_a.txt
"""

from pathlib import Path
import shutil
import re
from math import ceil
from collections import defaultdict
from loguru import logger


class FileSorterByIndex:
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        files_per_subfolder: int = 100,
        extensions: tuple = (".txt", ".csv", ".wav"),
    ):
        """
        :param source_dir: Directory with the original files
        :param output_dir: Directory to create sorted folders
        :param files_per_subfolder: Max number of files in each subfolder
        :param extensions: Tuple of file extensions to process
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.files_per_subfolder = files_per_subfolder
        self.extensions = tuple(ext.lower() for ext in extensions)

        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")

    @staticmethod
    def get_index(filename: str) -> str:
        """Extract numeric index from filename (before extension)."""
        name = Path(filename).stem
        match = re.search(r"(\d+)$", name)  # find digits at the end
        return match.group(1) if match else name

    def collect_files(self) -> dict:
        """Collect files by index from all subdirectories."""
        index_dict = defaultdict(list)

        # Use rglob to recursively find all files
        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                index = self.get_index(file_path.name)
                index_dict[index].append(file_path)

        logger.info(f"Found {len(index_dict)} unique indices")
        total_files = sum(len(files) for files in index_dict.values())
        logger.info(f"Total files to process: {total_files}")

        return index_dict

    def sort_files(self, copy_mode: bool = True):
        """
        Sort files into folders based on index and split into subfolders.

        :param copy_mode: If True, copy files. If False, move files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index_dict = self.collect_files()

        total_subfolders = 0
        total_files_copied = 0

        for index, files in index_dict.items():
            num_subfolders = ceil(len(files) / self.files_per_subfolder)

            for i in range(num_subfolders):
                # Create subfolder name (add suffix if multiple subfolders needed)
                if num_subfolders > 1:
                    subfolder_name = f"{index}_{i+1}"
                else:
                    subfolder_name = f"{index}"

                subfolder_path = self.output_dir / subfolder_name
                subfolder_path.mkdir(parents=True, exist_ok=True)

                start = i * self.files_per_subfolder
                end = start + self.files_per_subfolder

                for file_path in files[start:end]:
                    dest_path = subfolder_path / file_path.name

                    try:
                        if copy_mode:
                            shutil.copy2(file_path, dest_path)
                        else:
                            shutil.move(str(file_path), str(dest_path))
                        total_files_copied += 1
                    except Exception as e:
                        logger.error(f"Failed to process {file_path.name}: {e}")

                total_subfolders += 1
                logger.debug(f"Created {subfolder_name} with {len(files[start:end])} files")

        logger.info(f"Files sorted into {self.output_dir}")
        logger.info(f"Total subfolders created: {total_subfolders}")
        logger.info(f"Total files processed: {total_files_copied}")

    def get_statistics(self) -> dict:
        """Get statistics about the files to be sorted."""
        index_dict = self.collect_files()

        stats = {
            "total_indices": len(index_dict),
            "total_files": sum(len(files) for files in index_dict.values()),
            "files_by_extension": defaultdict(int),
            "indices_distribution": {},
        }

        for index, files in index_dict.items():
            stats["indices_distribution"][index] = len(files)
            for file_path in files:
                stats["files_by_extension"][file_path.suffix.lower()] += 1

        return stats


if __name__ == "__main__":
    sorter = FileSorterByIndex(
        source_dir="sentence_14/splits",
        output_dir="sentence_16/cleaned",
        files_per_subfolder=100,
        extensions=(".txt", ".csv", ".wav"),
    )

    # Optional: Print statistics before sorting
    stats = sorter.get_statistics()
    logger.info(f"Statistics: {stats}")

    # (copy mode by default) set to False to remove
    sorter.sort_files(copy_mode=True)
