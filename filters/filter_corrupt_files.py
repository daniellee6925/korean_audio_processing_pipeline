import asyncio
import concurrent.futures
import subprocess
from pathlib import Path
from loguru import logger
import os
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm


class FilterCorruptSegments:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_to_check = []
        self.AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".mp4", ".mkv")
        self.deleted_count = 0

    def collect_files(self):
        """Recursively collect all audio/video files in root_dir."""
        for root, dirs, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(self.AUDIO_EXTS):
                    full_path = os.path.join(root, f)
                    self.files_to_check.append(full_path)

    async def is_corrupt(self, file_path):
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-v",
            "warning",  # show warnings
            "-i",
            file_path,
            "-f",
            "null",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        logs = (stdout + stderr).decode("utf-8", errors="ignore")

        # Check for known corruption indicators
        corrupt_keywords = [
            "Malformed 'fmt '",
            "Invalid data found",
            "Reserved bit set",
            "Number of bands",
        ]
        for kw in corrupt_keywords:
            if kw in logs:
                return True
        return False

    async def delete_corrupt_file(self, file_path):
        """Delete the file if corrupt."""
        if await self.is_corrupt(file_path):
            # logger.info(f"Deleting corrupt file: {file_path}")
            os.remove(file_path)
            self.deleted_count += 1

    async def run(self, max_concurrent=100):
        self.collect_files()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def sem_task(f):
            async with semaphore:
                await self.delete_corrupt_file(f)
                return 1  # needed by tqdm_asyncio

        tasks = [sem_task(f) for f in self.files_to_check]

        await tqdm_asyncio.gather(
            *tasks,
            desc="Filtering Corrupt Segment Files",
            unit="file",
        )

    def process_all(self, max_concurrent=100):
        """Public method — runs the async process internally."""
        asyncio.run(self.run(max_concurrent))
        logger.success(f"Deleted {self.deleted_count} corrupted files")


class FilterCorrupt:
    def __init__(
        self,
        root_dir: str,
        max_workers: int = 10,
        delete_bad: bool = False,
    ):
        self.root_dir = root_dir
        self.max_workers = max_workers
        self.delete_bad = delete_bad
        self.AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".mp4", ".mkv")

    def is_corrupt(self, file_path: Path) -> bool:
        """Return True if FFmpeg fails to probe/decode the file."""
        try:
            # Use ffmpeg to check if file can be read
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(file_path), "-f", "null", "-"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return False
        except subprocess.CalledProcessError:
            return True

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
        files = [f for f in root.rglob("*") if f.suffix.lower() in self.AUDIO_EXTS]
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
        root_dir="poddbang_wavs/wavs_20250416_013301_segments",
        delete_bad=True,
    )
    filter.process_all()
