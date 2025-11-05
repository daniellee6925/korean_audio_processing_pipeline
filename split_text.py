import os
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)


def split_text_into_sentences(text: str) -> List[str]:
    """Split text file into sentneces"""
    sentences = sent_tokenize(text)
    return sentences


def main(root_dir: str = "archive") -> None:
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for text_file in sorted(os.listdir(folder_path)):
            if not text_file.lower().endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, text_file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        sentences = split_text_into_sentences(text)

        output_dir = os.path.join(folder_path, "text_sentences")
        os.makedirs(output_dir, exist_ok=True)
        for i, sentence in enumerate(sentences):
            outfile = os.path.join(output_dir, f"sentence_{i+1}.txt")
            with open(outfile, "w", encoding="utf-8") as out_f:
                out_f.write(sentence)
    print(f"Saved text_file for sentences in {root_dir}")


if __name__ == "__main__":
    main()
