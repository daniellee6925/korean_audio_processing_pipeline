import os
import yaml
import torch
import whisper
import ray
import numpy as np
import soundfile as sf
import time
from functools import wraps
from typing import Dict, Any, List, Tuple
from loguru import logger
from tqdm import tqdm
from pathlib import Path


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.success(f"Process Complete {func.__name__} runtime: {end - start:.2f} seconds")
        return result

    return wrapper


@ray.remote(num_gpus=0.25)
class WhisperBatchActor:
    """Ray actor that processes batches of audio files on GPU."""

    def __init__(
        self,
        model_name: str,
        transcribe_args: Dict[str, Any],
        use_fp16: bool = True,
        skip_long: bool = True,
    ):
        import torch
        import whisper

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcribe_args = transcribe_args.copy()
        self.use_fp16 = use_fp16
        self.model_name = model_name
        self.skip_long = skip_long
        self.model = None
        self.whisper = None

    def initialize_model(self) -> None:
        import torch
        import whisper

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.use_fp16 and self.device == "cuda":
            self.transcribe_args["fp16"] = True
            logger.debug("Using FP16 precision for faster inference")

        logger.debug(f"Actor loading Whisper model: {self.model_name} on {self.device}")
        self.model = whisper.load_model(self.model_name, device=self.device)
        self.whisper = whisper

    def load_and_pad_audio(self, audio_paths: List[str]) -> Tuple[torch.Tensor, list[int]]:
        """Load multiple audio files and pad to same length."""
        import torch

        audios = []
        padded_audios = []
        original_lengths = []

        for path in audio_paths:
            audio = self.whisper.load_audio(path)
            audios.append(audio)
            original_lengths.append(len(audio))

        max_length = max(original_lengths)

        # target_length = ((max_length - 1) // self.whisper.audio.SAMPLE_RATE + 1) * self.whisper.audio.SAMPLE_RATE
        target_length = 480000
        for audio in audios:
            if len(audio) < max_length:
                padded = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            else:
                padded = audio
            padded_audios.append(padded)

        batch_audio = torch.from_numpy(np.stack(padded_audios)).to(self.device)

        return batch_audio, original_lengths

    def get_audio_length(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            with sf.SoundFile(audio_path) as f:
                duration = len(f) / f.samplerate
                return duration
        except Exception as e:
            logger.error(f"Failed to load {audio_path} for length check: {e}")
            duration = 0.0

    def transcribe_batch(self, audio_paths: List[str]) -> List[Tuple[str, str, str]]:
        """Transcribe a batch of audio files together"""
        import torch

        results = []
        if self.skip_long:
            filtered_paths = []
            for path in audio_paths:
                if self.get_audio_length(path) > 30:
                    logger.warning(f"Skipping long file (>30s): {path}")
                    results.append((str(path), None, "skipped: duration > 30s"))
                else:
                    filtered_paths.append(path)
            audio_paths = filtered_paths

        # If all files were skipped, return early
        if not audio_paths:
            return results
        # try:
        #     # process batch through encoder once
        #     batch_audio, lengths = self.load_and_pad_audio(audio_paths)

        #     with torch.inference_mode():
        #         mel = self.whisper.log_mel_spectrogram(batch_audio).to(self.device)
        #     logger.debug(f"Processed batch of {len(audio_paths)} files into mel shape: {mel.shape}")

        #     if self.transcribe_args.get("language") is None:
        #         _, probs = self.model.detect_language(mel[0:1])  # [0:1] -> keep as batch
        #         detected_lang = max(probs, key=probs.get)
        #         logger.info(f"Detected language: {detected_lang}")
        #         self.transcribe_args["language"] = detected_lang

        #     options = self.whisper.DecodingOptions(**self.transcribe_args)
        #     with torch.inference_mode():
        #         for i, (path, mel_single) in enumerate(zip(audio_paths, mel)):
        #             decode_result = self.model.decode(mel_single.unsqueeze(0), options)

        #     for i, (path, result) in enumerate(zip(audio_paths, decode_results)):
        #         text = result.text.strip()
        #         results.append((path, text, None))

        # except Exception as e:
        #     logger.debug(f"Batch processing failed: {e}, falling back to individual processing")
        with torch.inference_mode():
            for path in audio_paths:
                try:
                    result = self.model.transcribe(path, **self.transcribe_args)
                    results.append((path, result["text"].strip(), None))
                except Exception as e2:
                    results.append((path, None, str(e2)))
        return results


class WhisperTranscriber:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self.load_config(config_path)

        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        processing_cfg = self.config["processing"]

        # ray configs
        self.num_gpus = processing_cfg.get("num_gpus", 1)
        self.num_actors = processing_cfg.get("num_actors", 4)
        self.batch_size = processing_cfg.get("batch_size", 8)
        self.ray_temp_dir = processing_cfg.get("ray_temp_dir", None)

        if not ray.is_initialized():
            ray.init(num_gpus=self.num_gpus, ignore_reinit_error=True, _temp_dir=self.ray_temp_dir)
            logger.info(f"Ray initialized with {self.num_gpus} GPUs")

        # file path configs
        self.root_dir = data_cfg.get("root_dir", "/workspace/audio_files")
        logger.info(f"Using root_dir: {self.root_dir}")
        self.recursive = data_cfg.get("recursive", False)
        self.extension = data_cfg.get("extension", ".wav")
        self.output_prefix = data_cfg.get("output_prefix", "")

        # Processing configs
        self.use_fp16 = processing_cfg.get("use_fp16", True)
        self.dynamic_batching = processing_cfg.get("dynamic_batching", False)
        self.skip_long = processing_cfg.get("skip_long", True)
        self.transcribe_args = self.config.get("transcription", {})

        # Create Ray actors
        logger.info(f"Creating {self.num_actors} Whisper actors...")
        self.actors = [
            WhisperBatchActor.remote(
                model_cfg["name"], self.transcribe_args, self.use_fp16, self.skip_long
            )
            for _ in range(self.num_actors)
        ]
        ray.get([actor.initialize_model.remote() for actor in self.actors])
        logger.info(f"Created {self.num_actors} actors with batch_size={self.batch_size}")

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """Load YAML config file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def find_audio_files(self) -> List[str]:
        """Find all audio files in directory (optionally recursive)."""
        root_path = Path(self.root_dir)

        if self.recursive:
            files = list(root_path.rglob(f"*.{self.extension}"))
        else:
            files = list(root_path.glob(f"*.{self.extension}"))
        return [str(f) for f in sorted(files)]

    def write_result(self, audio_path: str, text: str) -> None:
        """Write transcript txt file to corresponding folder location"""
        audio_path = Path(audio_path)
        out_path = audio_path.with_name(f"{self.output_prefix}{audio_path.stem}.txt")
        out_path.write_text(text, encoding="utf-8")

    def create_batches(self, audio_files: List[str]) -> List[List[str]]:
        """Create batches of audio files."""
        batches = []
        for i in range(0, len(audio_files), self.batch_size):
            batches.append(audio_files[i : i + self.batch_size])
        return batches

    def sort_by_duration(self, audio_files: List[str]) -> List[str]:
        logger.info("Analyzing audio file lengths for optimal batching...")
        file_durations = []
        for path in tqdm(audio_files, desc="Scanning Files for Dynamic Batching"):
            try:
                with sf.SoundFile(path) as f:
                    duration = len(f) / f.samplerate
                file_durations.append((path, duration))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                file_durations.append((path, 0))
        # sort by duration
        file_durations.sort(key=lambda x: x[1])
        sorted_files = [f[0] for f in file_durations]

        return sorted_files

    @timer
    def process_all(self) -> None:
        audio_files = self.find_audio_files()

        if not audio_files:
            logger.error(f"No audio files found in {self.root_dir}")
            return
        logger.info(f"Found {len(audio_files)} audio files in {self.root_dir}")

        # batch audio files
        if self.dynamic_batching:
            audio_files = self.sort_by_duration(audio_files)
        batches = self.create_batches(audio_files)
        logger.info(f"Split into {len(batches)} batches of size {self.batch_size}")

        # submit batch tasks to actors in round-robin
        pending_tasks = []
        actor_idx = 0

        for batch in batches:
            task = self.actors[actor_idx % len(self.actors)].transcribe_batch.remote(batch)
            pending_tasks.append(task)
            actor_idx += 1

        # process results as they are completed
        completed_files = 0
        with tqdm(total=len(audio_files), desc="Transcribing") as pbar:
            while pending_tasks:
                ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)

                for task in ready_tasks:
                    batch_results = ray.get(task)

                    for audio_path, text, error in batch_results:
                        if error:
                            logger.error(f"Failed to process {audio_path}: {error}")
                        else:
                            self.write_result(audio_path, text)

                        completed_files += 1
                        pbar.update()
        logger.info(f"Finished processing {len(audio_files)} files in {len(batches)} batches.")

    def shutdown(self):
        """Cleanup Ray resources."""
        ray.shutdown()
        if os.path.exists(self.ray_temp_dir):
            try:
                import shutil

                shutil.rmtree(self.ray_temp_dir)
                logger.info(f"Removed Ray temp dir: {self.ray_temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove Ray temp dir {self.ray_temp_dir}: {e}")
        logger.info("Ray shutdown complete")


if __name__ == "__main__":
    transcriber = WhisperTranscriber("configs/transcribe_config.yaml")
    try:
        transcriber.process_all()
    finally:
        transcriber.shutdown()
