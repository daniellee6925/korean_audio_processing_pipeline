import os
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)  # corrected from punkt_tab


class TextSentenceSplitter:
    def __init__(
        self,
        root_dir: str = "archive",
        output_folder_name: str = "text_sentences",
    ):
        """
        :param root_dir: Root directory to recursively scan for text files
        :param output_folder_name: Folder name where sentence files will be saved
        """
        self.root_dir = root_dir
        self.output_folder_name = output_folder_name

    @staticmethod
    def split_text_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)

    def process_text_file(self, file_path: str) -> None:
        """Split a single text file into sentences and save them."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = self.split_text_into_sentences(text)
        parent_dir = os.path.dirname(file_path)
        if os.path.basename(parent_dir) == self.output_folder_name:
            output_dir = parent_dir
        else:
            output_dir = os.path.join(parent_dir, self.output_folder_name)
            os.makedirs(output_dir, exist_ok=True)

        for i, sentence in enumerate(sentences):
            outfile = os.path.join(output_dir, f"sentence_{i+1}.txt")
            with open(outfile, "w", encoding="utf-8") as out_f:
                out_f.write(sentence)

    def process_directory(self) -> None:
        """Recursively process all text files under the root directory."""
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(".txt"):
                    file_path = os.path.join(dirpath, filename)
                    self.process_text_file(file_path)

        print(f"Saved sentence files for all text files in {self.root_dir}")


if __name__ == "__main__":
    splitter = TextSentenceSplitter(
        root_dir="sentence_14", output_folder_name="160101_014_Tr1_sentences"
    )
    splitter.process_directory()
