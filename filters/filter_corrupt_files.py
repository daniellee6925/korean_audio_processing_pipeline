import asyncio
import concurrent.futures
import subprocess
from pathlib import Path
from loguru import logger
import os

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".mp4"}


class FilterCorrupt:
    def __init__(
        self,
        root_dir: str,
        max_workers: int = 8,
        delete_bad: bool = False,
    ):
        self.root_dir = root_dir
        self.max_workers = max_workers
        self.delete_bad = delete_bad

    def probe_audio(self, file_path: Path) -> bool:
        """Check audio integrity using ffprobe without decoding the full file."""
        try:
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(file_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=True,
            )
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout: {file_path}")
        except subprocess.CalledProcessError:
            logger.error(f"Corrupted: {file_path}")
        except Exception as e:
            logger.error(f"Error checking {file_path}: {e}")
        return False

    async def check_all_audio(self):
        """Run ffprobe checks concurrently for all audio files, delete bad or non-WAV files."""
        root = Path(self.root_dir)
        files = [f for f in root.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
        logger.info(f"Found {len(files)} audio files under {self.root_dir}\n")

        bad_files = []

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Step 1: Check audio integrity
            tasks = [loop.run_in_executor(executor, self.probe_audio, f) for f in files]
            results = await asyncio.gather(*tasks)

        for f, ok in zip(files, results):
            if not ok:
                bad_files.append(f)
                if self.delete_bad:
                    try:
                        os.remove(f)
                        # logger.error(f"Deleted corrupted file: {f}")
                    except Exception as e:
                        logger.error(f"Could not delete {f}: {e}")

        logger.success(f"Finished Filtering - Total bad files: {len(bad_files)}")
        if not self.delete_bad:
            for f in bad_files:
                print(" -", f)

    def process_all(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return asyncio.ensure_future(self.check_all_audio())
        else:
            return asyncio.run(self.check_all_audio())


if __name__ == "__main__":
    filter = FilterCorrupt(
        root_dir="/Users/daniel/Desktop/projects/local_copy/wavs_20250416_012741",
        delete_bad=True,
        delete_non_wav=True,  # Delete mis-labeled WAVs
    )
    filter.process_all()
