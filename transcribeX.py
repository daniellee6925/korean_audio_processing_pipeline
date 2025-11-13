import os
import yaml
import torch
import whisperx
from typing import Dict, Any, List
from loguru import logger
from tqdm import tqdm


class WhisperXTranscriber:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)

        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        processing_config = self.config["processing"]

        self.device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = model_cfg.get(
            "compute_type", "float16" if self.device == "cuda" else "int8"
        )

        logger.info(
            f"Loading WhisperX model: {model_cfg['name']} on {self.device} with {self.compute_type}"
        )

        # WhisperX model loading
        self.model = whisperx.load_model(
            model_cfg["name"],
            device=self.device,
            compute_type=self.compute_type,
            language=model_cfg.get("language"),  # Optional: specify language for faster processing
        )

        # file path configs
        self.root_dir = data_cfg.get("root_dir", "/workspace/audio_files")
        self.recursive = data_cfg.get("recursive", False)
        self.extension = data_cfg.get("extension", ".wav")
        self.output_prefix = data_cfg.get("output_prefix", "out_")
        self.include_subdirs = data_cfg.get("include_subdirs", [])

        # WhisperX specific configs
        self.batch_size = processing_config.get("batch_size", 16)  # Higher = faster but more VRAM
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
            if self.include_subdirs:
                allowed_dirs = [os.path.join(self.root_dir, d) for d in self.include_subdirs]
            else:
                allowed_dirs = [self.root_dir]

            for allowed_dir in allowed_dirs:
                if not os.path.exists(allowed_dir):
                    logger.warning(f"Directory not found: {allowed_dir}")
                    continue
                for dirpath, _, filenames in os.walk(allowed_dir):
                    for f in filenames:
                        if f.lower().endswith(self.extension):
                            files.append(os.path.join(dirpath, f))
        return sorted(files)

    def transcribe_file(self, wav_path: str) -> str:
        """Transcribe a single audio file using WhisperX."""
        # Load audio
        audio = whisperx.load_audio(wav_path)

        # Transcribe with batching
        result = self.model.transcribe(audio, batch_size=self.batch_size, **self.transcribe_args)

        # Extract text from segments
        text = " ".join([segment["text"].strip() for segment in result["segments"]])
        return text.strip()

    def process_all(self) -> None:
        """Process all audio files under root_dir using WhisperX with a progress bar."""
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
    transcriber = WhisperXTranscriber("config.yaml")
    transcriber.process_all()
