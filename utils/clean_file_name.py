from pathlib import Path
from loguru import logger
from typing import Union, List


class CleanFileName:
    def __init__(
        self,
        root_dir: str,
        extensions: Union[str, List[str]],
        original: str,
        change_to: str,
    ):
        self.root_dir = Path(root_dir)
        # Normalize extensions to a list
        if isinstance(extensions, str):
            self.extensions = [extensions]
        else:
            self.extensions = extensions

        self.original = original
        self.change_to = change_to

        if not self.root_dir.exists():
            raise ValueError(f"Folder does not exist: {root_dir}")

    def clean_segment_prefix(self, file_path: Path):
        """Rename a single file if it contains the original substring."""
        if self.original in file_path.name:
            new_name = file_path.name.replace(self.original, self.change_to)
            new_path = file_path.with_name(new_name)
            file_path.rename(new_path)
            logger.info(f"Renamed: {file_path} → {new_path}")

    def process_all(self):
        """Process all files under the root directory recursively for all extensions."""
        files = []
        for ext in self.extensions:
            files.extend(self.root_dir.rglob(f"*.{ext}"))

        if not files:
            logger.info(f"No files found under {self.root_dir} for extensions {self.extensions}")
            return

        for file in files:
            self.clean_segment_prefix(file)

        logger.info(f"Finished processing {len(files)} files.")


if __name__ == "__main__":
    cleaner = CleanFileName(
        "audio_files_2_sentences",
        extensions=["txt"],  # can process multiple types
        original="transcribed_",
        change_to="",
    )
    cleaner.process_all()
