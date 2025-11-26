import torch
import numpy as np
import yaml
import subprocess
import io
import torchaudio
from typing import Protocol, Iterator, Any, List, Tuple, Optional
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from faster_whisper import WhisperModel
from dataclasses import dataclass, field
import time


class TranscriptionModel(Protocol):
    """Protocol for any model that can transcribe audio"""

    def transcribe(
        self,
        audio: np.array,
        language: Optional[str] = None,
        beam_size: int = 1,
        best_of: int = 1,
        vad_filter: bool = False,
        word_timestamps: bool = False,
    ) -> Tuple[Iterator[Any], Any]:
        """Transcribe audio and return segment + info"""
        ...


class AudioLoader:
    """Handles audio file I/O operations"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio_ffmpeg(self, audio: bytes) -> np.array:
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-sr",
                str(self.sample_rate),
                "-",
            ]
            out = subprocess.run(cmd, input=audio, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def load_audio_torchaudio(self, audio: bytes) -> np.ndarray:
        data_or_path_or_list, audio_fs = torchaudio.load(io.BytesIO(audio))
        data_or_path_or_list = data_or_path_or_list.mean(0)
        if audio_fs != self.sample_rate:
            resampler = torchaudio.transforms.Resample(audio_fs, self.sample_rate)
            data_or_path_or_list = resampler(data_or_path_or_list)
        return data_or_path_or_list.numpy()

    def load_audio(
        self,
        audio: bytes,
    ) -> np.ndarray:
        try:
            return self.load_audio_torchaudio(audio)
        except:
            return self.load_audio_ffmpeg(audio)


@dataclass
class ModelConfig:
    name: str = "large-v3"
    device: str = "cuda"


@dataclass
class DataConfig:
    root_dir: str = "/workspace/audio_files"
    recursive: bool = True
    extension: str = ".wav"
    output_prefix: str = ""


@dataclass
class ProcessingConfig:
    compute_type: str = "float16"
    device_id: int = 0
    num_workers: int = 6


@dataclass
class TranscriptionConfig:
    language: Optional[str] = None
    beam_size: int = 1  # 1 = greedy, much faster
    best_of: int = 1
    vad_filter: bool = False
    word_timestamps: bool = False


@dataclass
class TranscribeConfig:
    """Comoplete configuration for transcription"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "TranscribeConfig":
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get("Model", {})),
            data=DataConfig(**config_dict.get("Data", {})),
            processing=ProcessingConfig(**config_dict.get("processing", {})),
            transcription=TranscriptionConfig(**config_dict.get("transcription", {})),
        )


class Transcriber:
    """Transcriber with multiple workers on single GPU for parallel processing"""

    def __init__(self, config_path: str):
        self.config = TranscribeConfig.from_yaml(config_path)

        # Optimize CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.models: List[TranscriptionModel] = []
        self.initialize_models()
        self.audio_loader = AudioLoader()

        logger.info(f"Ready with {self.config.processing.num_workers} workers")

    def initialize_models(self) -> None:
        """Initialize multiple  model instances on the same GPU."""
        logger.info(
            f"Loading {self.config.processing.num_workers} x {self.config.model.name} models on GPU {self.config.processing.device_id}..."
        )

        for i in range(self.config.processing.num_workers):
            model = WhisperModel(
                self.config.model.name,
                device=self.config.model.device,
                device_index=self.config.processing.device_id,
                compute_type=self.config.processing.compute_type,
                num_workers=1,  # internal workers per model
            )
            self.models.append(model)
            logger.info(f"Worker {i+1}/{self.config.processing.num_workers} loaded")

        gpu_mem = torch.cuda.memory_allocated(self.config.processing.device_id) / 1024**3
        logger.info(f"Total GPU memory used: {gpu_mem:.2f} GB")

    def find_audio_files(self) -> List[str]:
        """find all audio files in directory"""
        root_path = Path(self.config.data.root_dir)
        if self.config.data.recursive:
            files = list(root_path.rglob(f"*.{self.config.data.extension}"))
        else:
            files = list(root_path.glob(f"*.{self.config.data.extension}"))

        unprocessed = []
        for f in sorted(files):
            txt_path = f.with_name(f"{self.config.data.output_prefix}{f.stem}.txt")
            if not txt_path.exists():
                unprocessed.append(str(f))

        logger.info(f"Found {len(files)} total files, {len(unprocessed)} unprocessed files")
        return unprocessed

    def write_result(self, audio_path: str, text: str) -> None:
        """Write transcript to txt file"""
        audio_path = Path(audio_path)
        out_path = audio_path.with_name(f"{self.config.data.output_prefix}{audio_path.stem}.txt")
        out_path.write_text(text, encoding="utf-8")

    def load_audio_files(self, audio_path: str) -> Tuple[str, Optional[np.array], Optional[str]]:
        """Load AUdio file (I/O) operation"""
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio = self.audio_loader.load_audio(audio_bytes)
            return audio_path, audio, None
        except Exception as e:
            return audio_path, None, str(e)

    def transcribe_file(
        self, model: TranscriptionModel, audio_path: str, audio: np.ndarray
    ) -> Tuple[str, bool, Optional[str]]:
        """Transcribe a single audio file with a given model"""
        try:
            segments, info = model.transcribe(
                audio,
                language=self.config.transcription.language,
                beam_size=self.config.transcription.beam_size,
                best_of=self.config.transcription.best_of,
                vad_filter=self.config.transcription.vad_filter,
                word_timestamps=self.config.transcription.word_timestamps,
            )

            text = " ".join((seg.text.strip()) for seg in segments)
            self.write_result(audio_path, text)
            return audio_path, True, None
        except Exception as e:
            return audio_path, False, str(e)

    def cleanup(self):
        """clean up resources"""
        logger.info("Cleaning up models...")
        for model in self.models:
            del model
        self.models.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"GPU memory after cleanup: {torch.cuda.memory_allocated(self.config.processing.device_id) / 1024**3:.2f} GB"
            )

    def process_all(self) -> None:
        """Process all audio files using multiple workers"""
        audio_files = self.find_audio_files()

        if not audio_files:
            logger.error(f"No audio files found in {self.config.data.root_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files")
        logger.info(f"Processing with {self.config.processing.num_workers} parallel workers...")

        total_files, error_count = 0, 0
        start_time = time.time()

        # parallel processing
        io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="IO")
        gpu_executor = ThreadPoolExecutor(
            max_workers=self.config.processing.num_workers, thread_name_prefix="GPU"
        )
        try:
            io_futures = {}
            for audio_path in audio_files:
                future = io_executor.submit(self.load_audio_files, audio_path)
                io_futures[future] = audio_path

            gpu_futures = {}
            model_idx = 0

            with tqdm(total=len(audio_files), desc="transcribing", unit="files") as pbar:
                for io_future in as_completed(io_futures):
                    audio_path = io_futures[io_future]

                    try:
                        path, audio, error = io_future.result()
                        if audio is None:
                            logger.error(f"Failed {path}: {error}")
                            error_count += 1
                            total_files += 1
                            pbar.update(1)
                            continue

                        model = self.models[model_idx % self.config.processing.num_workers]
                        gpu_future = gpu_executor.submit(self.transcribe_file, model, path, audio)
                        gpu_futures[gpu_future] = path
                        model_idx += 1
                    except Exception as e:
                        logger.error(f"Exception processing {audio_path}: {e}")
                        error_count += 1
                        total_files += 1
                        pbar.update(1)
                for gpu_future in as_completed(gpu_futures):
                    audio_path = gpu_futures[gpu_future]

                    try:
                        path, success, error = gpu_future.result()
                        if not success:
                            logger.error(f"Failed to transcribe {path}: {error}")
                            error_count += 1
                        total_files += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Exception processing {audio_path}: {e}")
                        error_count += 1
                        total_files += 1
                        pbar.update(1)

        except KeyboardInterrupt:
            logger.warning("\nKeyboardInterrupt received! Shutting down gracefully...")
            # Cancel pending futures
            for future in list(io_futures.keys()) + list(gpu_futures.keys()):
                future.cancel()

        finally:
            # Always shutdown executor and wait for threads to finish
            logger.info("Shutting down executor...")
            io_executor.shutdown(wait=True)
            gpu_executor.shutdown(wait=True)
            logger.info("All threads closed.")

            elapsed = time.time() - start_time
            avg_speed = total_files / elapsed if elapsed > 0 else 0

            logger.info("=" * 70)
            logger.info(f"FINISHED!")
            logger.info(
                f"Total files processed: {total_files}/{len(audio_files)} ({error_count} errors)"
            )
            logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
            logger.info(f"Average speed: {avg_speed:.1f} files/second")
            logger.info(
                f"Final GPU memory: {torch.cuda.memory_allocated(self.config.processing.device_id) / 1024**3:.1f} GB"
            )
            logger.info("=" * 70)


if __name__ == "__main__":
    transcriber = Transcriber("configs/faster_transcriber_config.yaml")
    transcriber.process_all()
