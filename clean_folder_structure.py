import os
import shutil
from math import ceil
from collections import defaultdict


class FileSorterByIndex:
    def __init__(
        self, source_dir: str, output_dir: str, files_per_subfolder: int = 100
    ):
        """
        :param source_dir: Directory with the original files
        :param output_dir: Directory to create sorted folders
        :param files_per_subfolder: Max number of files in each subfolder
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.files_per_subfolder = files_per_subfolder
        self.extensions = (".txt", ".csv", ".wav")

    @staticmethod
    def get_index(filename: str) -> str:
        """Get the last 1-2 characters of the filename (before extension)."""
        name, _ = os.path.splitext(filename)
        return name[-2:] if len(name) > 1 else name[-1]

    def collect_files(self) -> dict:
        """Collect files by index."""
        index_dict = defaultdict(list)
        for root, _, files in os.walk(self.source_dir):
            for f in files:
                if f.lower().endswith(self.extensions):
                    index = self.get_index(f)
                    index_dict[index].append(os.path.join(root, f))
        return index_dict

    def sort_files(self):
        """Sort files into folders based on index and split into subfolders."""
        os.makedirs(self.output_dir, exist_ok=True)
        index_dict = self.collect_files()

        for index, files in index_dict.items():
            num_subfolders = ceil(len(files) / self.files_per_subfolder)
            for i in range(num_subfolders):
                subfolder_path = os.path.join(self.output_dir, f"{index}")
                os.makedirs(subfolder_path, exist_ok=True)
                start = i * self.files_per_subfolder
                end = start + self.files_per_subfolder
                for file_path in files[start:end]:
                    shutil.copy(file_path, subfolder_path)
        print(f"Files sorted into {self.output_dir}")


if __name__ == "__main__":
    sorter = FileSorterByIndex(
        source_dir="sentence_14/splits",
        output_dir="sentence_16/cleaned",
        files_per_subfolder=50,
    )
    sorter.sort_files()
