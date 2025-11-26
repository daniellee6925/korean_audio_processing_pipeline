from tn.korean.normalizer import Normalizer as KoNormalizer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Union, List
from loguru import logger
import shutil
import time


class KoreanTextNormalizer:
    def __init__(self, overwrite_cache: bool = False, max_workers: int = 8, override: bool = False):
        """
        Wrapper for the WeTextProcessingBoson Korean Normalizer.
        """
        self.model = KoNormalizer(overwrite_cache=overwrite_cache)
        self.max_workers = max_workers
        self.override = override

    def normalize_text(self, text: str) -> str:
        """
        Normalize a single string.
        """
        return self.model.normalize(text)

    def normalize_list(self, texts: List[str]) -> List[str]:
        """
        Normalize a list of strings.
        """
        return [self.model.normalize(t) for t in texts]

    def normalize_file(self, input_path: Union[str, Path], output_path: Union[str, Path] = None):
        """
        Normalize a text file line by line and save the output.
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
        else:
            output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        with input_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            print(lines)

        normalized_lines = [self.model.normalize(line.strip()) + "\n" for line in lines]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.writelines(normalized_lines)

        if self.override:
            # Replace the original file with temp file
            shutil.move(str(output_path), str(input_path))

        logger.info(f"Normalized file saved to: {output_path}")

    def normalize_all(self, root_dir: Union[str, Path]):
        """
        Recursively find and normalize all .txt files under root_dir.
        Saves each normalized file in the same folder with '_normalized' suffix.
        """
        root_dir = Path(root_dir)
        all_txt_files = list(root_dir.rglob("*.txt"))
        logger.info(f"Found {len(all_txt_files)} text files under {root_dir}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.normalize_file, txt_file): txt_file
                for txt_file in all_txt_files
            }
            for future in futures:
                txt_file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error {e} normalizing {txt_file}")


if __name__ == "__main__":
    # Example usage
    normalizer = KoreanTextNormalizer(overwrite_cache=False, max_workers=8, override=False)
    start = time.time()
    # normalizer.normalize_all("Results_Text")
    normalizer.normalize_file(input_path="output_unique.txt", output_path="output.txt")
    end = time.time()

    logger.success(f"job completed in {end-start:.2f} seconds")
