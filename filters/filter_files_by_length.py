import os
import soundfile as sf
from loguru import logger
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class FilterFiles:
    def __init__(
        self,
        root_dir: str,
        file_format: str = "wav",
        min_dur: float = None,
        max_dur: float = None,
        max_workers: int = 8,
        use_multithread: bool = True,
    ):
        """
        Args:
            root_dir (str): Path to the root directory
            file_format (str): e.g., "wav", "mp3", "flac". Default is "wav".
            min_dur (float, optional): Minimum duration in seconds. Files shorter than this are filtered.
            max_dur (float, optional): Maximum duration in seconds. Files longer than this are filtered.
            max_workers (int): Number of threads for multithreaded deletion.
            use_multithread (bool): Whether to use multithreading for file deletion.
        """
        self.root_dir = root_dir
        self.file_format = file_format.lower()
        self.min_duration = min_dur
        self.max_duration = max_dur
        self.max_workers = max_workers
        self.use_multithread = use_multithread

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Returns duration in seconds for an audio file. None if unreadable."""
        try:
            with sf.SoundFile(file_path) as f:
                duration = len(f) / f.samplerate
            return duration
        except Exception as e:
            logger.error(f"Failed to get duration for {file_path}: {e}")
            return None

    def filter_by_duration(self, file_list: List[str]) -> List[str]:
        """Filters a list of audio files by min and max duration in seconds."""
        remove_files = []
        for f in file_list:
            duration = self.get_audio_duration(f)
            if duration is None:
                continue
            if self.min_duration is not None and duration < self.min_duration:
                remove_files.append(f)
                continue
            if self.max_duration is not None and duration > self.max_duration:
                remove_files.append(f)
                continue
        logger.info(f"Detected {len(remove_files)} files to remove in this batch.")
        return remove_files

    def process_all(self) -> None:
        """Walks the root directory and removes filtered files."""
        total_deleted = 0

        def delete_file(f: str) -> bool:
            try:
                os.remove(f)
                return True
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
                return False

        for dirpath, _, files in os.walk(self.root_dir):
            audio_files = [
                os.path.join(dirpath, f)
                for f in files
                if f.lower().endswith(f".{self.file_format}")
            ]
            if not audio_files:
                continue

            logger.info(f"Filtering {len(audio_files)} files in {dirpath}")
            files_to_remove = self.filter_by_duration(audio_files)

            if files_to_remove:
                if self.use_multithread and len(files_to_remove) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {executor.submit(delete_file, f): f for f in files_to_remove}
                        for future in tqdm(
                            as_completed(futures), total=len(futures), desc=f"Deleting in {dirpath}"
                        ):
                            if future.result():
                                total_deleted += 1
                else:
                    for f in tqdm(files_to_remove, desc=f"Deleting in {dirpath}"):
                        if delete_file(f):
                            total_deleted += 1

                logger.info(f"Removed {len(files_to_remove)} files in {dirpath}")
            else:
                logger.info(f"No files to remove in {dirpath}")

        logger.success(f"Finished filtering. Total number of files deleted: {total_deleted}")


# ----------------- Usage Example -----------------
if __name__ == "__main__":
    filter = FilterFiles(
        root_dir="data/wavs_20250416_012741_splits",
        file_format="wav",
        min_dur=5.0,  # sec
        max_dur=10.0,
        max_workers=8,
        use_multithread=True,  # Set to False to disable multithreading
    )
    filter.process_all()
