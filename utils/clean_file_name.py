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
        portion: str = "any",  # "start", "any", or "custom"
        custom_range: tuple[int, int] = None,  # only used if portion="custom"
    ):
        self.root_dir = Path(root_dir)
        if isinstance(extensions, str):
            self.extensions = [extensions]
        else:
            self.extensions = extensions

        self.original = original
        self.change_to = change_to
        self.portion = portion
        self.custom_range = custom_range

        if not self.root_dir.exists():
            raise ValueError(f"Folder does not exist: {root_dir}")

    def clean_segment_prefix(self, file_path: Path):
        name = file_path.stem
        ext = file_path.suffix

        new_name = name  # default, no change

        if self.portion == "start":
            if name.startswith(self.original):
                new_name = self.change_to + name[len(self.original) :]

        elif self.portion == "custom" and self.custom_range:
            start, end = self.custom_range
            portion_str = name[start:end]
            if self.original in portion_str:
                portion_str = portion_str.replace(self.original, self.change_to)
                new_name = name[:start] + portion_str + name[end:]

        else:  # default "any"
            if self.original in name:
                new_name = name.replace(self.original, self.change_to)

        if new_name != name:
            new_path = file_path.with_name(new_name + ext)
            file_path.rename(new_path)
            logger.info(f"Renamed: {file_path} → {new_path}")

    def process_all(self):
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
        "TEXT_251111_1",
        extensions=["txt"],
        original="1_",
        change_to="",
        portion="start",  # only replace at the start
        # portion="custom", custom_range=(0,5)  # uncomment for a custom portion
    )
    cleaner.process_all()
