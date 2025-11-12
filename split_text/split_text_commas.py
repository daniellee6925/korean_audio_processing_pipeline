import os
import re
from typing import List


class TextSentenceSplitter:
    def __init__(
        self,
        root_dir: str = "archive",
        output_suffix: str = "_segment",
        sentence_folders: bool = True,
    ):
        """
        :param root_dir: Root directory to recursively scan for text files
        :param output_suffix: Suffix added to create a sentence folder for each text file
        :param sentence_folders: If True, each sentence will be stored inside its own folder
        """
        self.root_dir = root_dir
        self.output_suffix = output_suffix
        self.sentence_folders = sentence_folders

    @staticmethod
    def split_text_into_sentences(text: str) -> List[str]:
        """Split text into sentences on periods and commas."""
        # Split on periods or commas, keep non-empty stripped parts
        sentences = [s.strip() for s in re.split(r"[.,]", text) if s.strip()]
        return sentences

    def process_text_file(self, file_path: str) -> None:
        """Split a single text file into sentences and save them."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = self.split_text_into_sentences(text)

        # Create a folder for this text file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), f"{base_name}{self.output_suffix}")
        os.makedirs(output_dir, exist_ok=True)

        for i, sentence in enumerate(sentences, start=1):
            if self.sentence_folders:
                # Create subfolder for each sentence
                sentence_dir = os.path.join(output_dir, f"segment_{i}")
                os.makedirs(sentence_dir, exist_ok=True)
                outfile = os.path.join(sentence_dir, f"segment_{i}.txt")
            else:
                # Save all sentence files directly in output_dir
                outfile = os.path.join(output_dir, f"segment_{i}.txt")

            with open(outfile, "w", encoding="utf-8") as out_f:
                out_f.write(sentence)

    def process_directory(self) -> None:
        """Recursively process all text files under the root directory."""
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(".txt"):
                    file_path = os.path.join(dirpath, filename)
                    self.process_text_file(file_path)

        print(f"Created sentence folders for all text files in '{self.root_dir}'")


if __name__ == "__main__":
    splitter = TextSentenceSplitter(root_dir="1107_Recording_sentences", sentence_folders=True)
    splitter.process_directory()
