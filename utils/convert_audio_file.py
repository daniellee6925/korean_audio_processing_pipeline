import os
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed


class AudioConverter:
    def __init__(
        self,
        folder_path: str,
        output_format: str,
        remove_org: bool = False,
        use_multiprocessing: bool = False,
        max_workers: int = None,
    ):
        """
        Args:
            folder_path (str): Root folder containing audio files to convert
            output_format (str): Desired format, e.g., "wav", "mp3", "flac"
            remove_org (bool): If True, original files are deleted after conversion
            use_multiprocessing (bool): If True, convert files in parallel
            max_workers (int): Number of processes for multiprocessing (None = auto)
        """
        self.folder_path = Path(folder_path)
        self.output_format = output_format.lower()
        self.remove_org = remove_org
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers

    def convert_file(self, input_file: str):
        """Converts a single audio file to the specified format."""
        try:
            audio = AudioSegment.from_file(input_file)
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.{self.output_format}"
            audio.export(output_file, format=self.output_format)
            if self.remove_org:
                os.remove(input_file)
            return input_file, output_file, None
        except Exception as e:
            return input_file, None, str(e)

    def process_all(self):
        """Recursively converts all audio files in folder_path to output_format."""
        audio_files = list(self.folder_path.rglob("*.*"))
        audio_files = [str(f) for f in audio_files if f.suffix.lower() in [".wav", ".mp3", ".flac"]]

        print(f"Found {len(audio_files)} audio files in {self.folder_path}")

        if not audio_files:
            return

        if self.use_multiprocessing:
            # Process files in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.convert_file, f): f for f in audio_files}
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Converting audio files"
                ):
                    input_file, output_file, error = future.result()
                    if error:
                        print(f"[ERROR] Failed: {input_file} → {error}")
        else:
            # Process files sequentially
            for file_path in tqdm(audio_files, desc="Converting audio files"):
                input_file, output_file, error = self.convert_file(file_path)
                if error:
                    print(f"[ERROR] Failed: {input_file} → {error}")


# ----------------- Usage -----------------
if __name__ == "__main__":
    converter = AudioConverter(
        folder_path="audio_files_2",
        output_format="flac",
        remove_org=False,
        use_multiprocessing=True,  # set to False to disable
        max_workers=8,  # number of parallel processes
    )
    converter.process_all()
