import os
import yaml
import torch
import whisper
from typing import Dict, Any, List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class WhisperTranscriber:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)

        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        processing_config = self.config["processing"]

        self.device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Whisper model: {model_cfg['name']} on {self.device}")
        # model configs
        self.model = whisper.load_model(model_cfg["name"], device=self.device)

        # file path configs
        self.root_dir = data_cfg.get("root_dir", "/workspace/audio_files")
        self.recursive = data_cfg.get("recursive", False)
        self.extension = data_cfg.get("extension", ".wav")
        self.output_prefix = data_cfg.get("output_prefix", "out_")
        self.max_workers = processing_config.get("max_workers", 4)

        self.transcribe_args = self.config.get("transcription", {})

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def find_audio_files(self, root_dir: str) -> List[str]:
        """Find all audio files in directory (optionally recursive)."""
        files = []

        if self.recursive:
            for dirpath, _, filenames in os.walk(root_dir):
                for f in filenames:
                    if f.lower().endswith(self.extension):
                        files.append(os.path.join(dirpath, f))
        else:
            allowed_dirs = [os.path.join(self.root_dir, d) for d in self.include_subdirs]
            for allowed_dir in allowed_dirs:
                for dirpath, _, filenames in os.walk(allowed_dir):
                    for f in filenames:
                        if f.lower().endswith(self.extension):
                            files.append(os.path.join(dirpath, f))
        return sorted(files)

    def transcribe_file(self, wav_path: str) -> str:
        """Transcribe a single audio file."""
        result = self.model.transcribe(wav_path, **self.transcribe_args)
        return result["text"].strip()

    def process_all(self) -> None:
        """Process all audio files under root_dir (recursively) using Whisper with a progress bar."""
        audio_files = self.find_audio_files(self.root_dir)

        if not audio_files:
            logger.error(f"No audio files found in {self.root_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files in {self.root_dir}")

        for wav_path in tqdm(audio_files, total=len(audio_files), desc="Transcribing"):
            try:
                text = self.transcribe_file(wav_path)
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                out_path = os.path.join(
                    os.path.dirname(wav_path),
                    f"{self.output_prefix}{base_name}.txt",
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                logger.error(f"Failed to process {wav_path}: {e}")

        logger.info(f"Finished processing {len(audio_files)} files.")


if __name__ == "__main__":
    transcriber = WhisperTranscriber("config.yaml")
    transcriber.process_all()
