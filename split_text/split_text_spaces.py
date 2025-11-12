import os
from typing import List


class TextLineSplitter:
    def __init__(
        self,
        root_dir: str = "archive",
        output_suffix: str = "_segment",
        line_folders: bool = True,
    ):
        """
        :param root_dir: Root directory to recursively scan for text files
        :param output_suffix: Suffix added to create a line folder for each text file
        :param line_folders: If True, each line will be stored inside its own folder
        """
        self.root_dir = root_dir
        self.output_suffix = output_suffix
        self.line_folders = line_folders

    @staticmethod
    def split_text_by_newline(text: str) -> List[str]:
        """Split text into non-empty lines."""
        return [line.strip() for line in text.splitlines() if line.strip()]

    def process_text_file(self, file_path: str) -> None:
        """Split a single text file by newlines and save each line."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = self.split_text_by_newline(text)

        # Create a folder for this text file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), f"{base_name}{self.output_suffix}")
        os.makedirs(output_dir, exist_ok=True)

        for i, line in enumerate(lines, start=1):
            if self.line_folders:
                # Create subfolder for each line
                line_dir = os.path.join(output_dir, f"segment_{i}")
                os.makedirs(line_dir, exist_ok=True)
                outfile = os.path.join(line_dir, f"segment_{i}.txt")
            else:
                # Save all line files directly in output_dir
                outfile = os.path.join(output_dir, f"segment_{i}.txt")

            with open(outfile, "w", encoding="utf-8") as out_f:
                out_f.write(line)

    def process_directory(self) -> None:
        """Recursively process all text files under the root directory."""
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(".txt"):
                    file_path = os.path.join(dirpath, filename)
                    self.process_text_file(file_path)

        print(f"Created line folders for all text files in '{self.root_dir}'")


if __name__ == "__main__":
    splitter = TextLineSplitter(
        root_dir="TEXT_251111", line_folders=True  # toggle this to True/False
    )
    splitter.process_directory()
