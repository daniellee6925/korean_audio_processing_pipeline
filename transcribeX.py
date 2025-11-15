import os
import yaml
import torch
import whisper
import ray
from typing import Dict, Any, List
from loguru import logger
from tqdm import tqdm


@ray.remote(num_gpus=1)
class WhisperActor:
    """Ray actor that holds a Whisper model on GPU."""

    def __init__(self, model_name: str, transcribe_args: Dict[str, Any], use_fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Enable A100 optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        logger.info(f"Loading Whisper model: {model_name} on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)

        # Enable fp16 for A100
        self.use_fp16 = use_fp16
        if self.use_fp16 and self.device == "cuda":
            self.model = self.model.half()
            logger.info("Using FP16 precision for faster inference")

        self.transcribe_args = transcribe_args.copy()
        if self.use_fp16 and self.device == "cuda":
            self.transcribe_args["fp16"] = True

    def transcribe(self, wav_path: str) -> tuple:
        """Transcribe a single audio file."""
        try:
            with torch.inference_mode():
                result = self.model.transcribe(wav_path, **self.transcribe_args)
            return wav_path, result["text"].strip(), None
        except Exception as e:
            return wav_path, None, str(e)


class WhisperTranscriber:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)

        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        processing_config = self.config["processing"]

        # Ray configuration
        self.num_gpus = processing_config.get("num_gpus", 1)
        self.num_actors = processing_config.get("num_actors", 1)

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_gpus=self.num_gpus,
                ignore_reinit_error=True,
                _temp_dir=processing_config.get("ray_temp_dir", None),
            )
            logger.info(f"Ray initialized with {self.num_gpus} GPUs")

        # File path configs
        self.root_dir = data_cfg.get("root_dir", "/workspace/audio_files")
        logger.info(f"Using root_dir: {self.root_dir}")
        self.recursive = data_cfg.get("recursive", False)
        self.extension = data_cfg.get("extension", ".wav")
        self.output_prefix = data_cfg.get("output_prefix", "out_")

        # Processing configs
        self.use_fp16 = processing_config.get("use_fp16", True)
        self.prefetch_size = processing_config.get("prefetch_size", 10)

        self.transcribe_args = self.config.get("transcription", {})

        # Create Ray actors (one per GPU fraction)
        logger.info(f"Creating {self.num_actors} Whisper actors...")
        self.actors = [
            WhisperActor.remote(model_cfg["name"], self.transcribe_args, self.use_fp16)
            for _ in range(self.num_actors)
        ]
        logger.info(f"Created {self.num_actors} actors")

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
            for f in os.listdir(root_dir):
                if f.lower().endswith(self.extension):
                    files.append(os.path.join(root_dir, f))

        return sorted(files)

    def write_result(self, wav_path: str, text: str) -> None:
        """Write transcription result to file."""
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        out_path = os.path.join(
            os.path.dirname(wav_path),
            f"{self.output_prefix}{base_name}.txt",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    def process_all(self) -> None:
        """Process all audio files using Ray actors with load balancing."""
        audio_files = self.find_audio_files(self.root_dir)

        if not audio_files:
            logger.error(f"No audio files found in {self.root_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files in {self.root_dir}")

        # Submit tasks in round-robin fashion to actors
        pending_tasks = []
        actor_idx = 0

        # Submit initial batch
        for wav_path in audio_files[: self.prefetch_size]:
            task = self.actors[actor_idx % len(self.actors)].transcribe.remote(wav_path)
            pending_tasks.append(task)
            actor_idx += 1

        # Process remaining files with pipeline
        next_file_idx = self.prefetch_size
        completed = 0

        with tqdm(total=len(audio_files), desc="Transcribing") as pbar:
            while pending_tasks:
                # Wait for at least one task to complete
                ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)

                for task in ready_tasks:
                    wav_path, text, error = ray.get(task)

                    if error:
                        logger.error(f"Failed to process {wav_path}: {error}")
                    else:
                        self.write_result(wav_path, text)

                    completed += 1
                    pbar.update(1)

                    # Submit next task to keep pipeline full
                    if next_file_idx < len(audio_files):
                        next_task = self.actors[actor_idx % len(self.actors)].transcribe.remote(
                            audio_files[next_file_idx]
                        )
                        pending_tasks.append(next_task)
                        actor_idx += 1
                        next_file_idx += 1

        logger.info(f"Finished processing {len(audio_files)} files.")

    def process_all_batch(self, batch_size: int = None) -> None:
        """
        Alternative: Process in larger batches for maximum throughput.
        Better when you have many files and want maximum GPU utilization.
        """
        audio_files = self.find_audio_files(self.root_dir)

        if not audio_files:
            logger.error(f"No audio files found in {self.root_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files in {self.root_dir}")

        # Default batch size: 2-3x number of actors
        if batch_size is None:
            batch_size = len(self.actors) * 3

        # Submit all tasks
        all_tasks = []
        actor_idx = 0

        for wav_path in audio_files:
            task = self.actors[actor_idx % len(self.actors)].transcribe.remote(wav_path)
            all_tasks.append(task)
            actor_idx += 1

        # Process results as they complete
        with tqdm(total=len(audio_files), desc="Transcribing") as pbar:
            while all_tasks:
                # Wait for a batch of tasks
                ready_tasks, all_tasks = ray.wait(
                    all_tasks, num_returns=min(batch_size, len(all_tasks))
                )

                for task in ready_tasks:
                    wav_path, text, error = ray.get(task)

                    if error:
                        logger.error(f"Failed to process {wav_path}: {error}")
                    else:
                        self.write_result(wav_path, text)

                    pbar.update(1)

        logger.info(f"Finished processing {len(audio_files)} files.")

    def shutdown(self):
        """Cleanup Ray resources."""
        ray.shutdown()
        logger.info("Ray shutdown complete")


if __name__ == "__main__":
    transcriber = WhisperTranscriber("transcribe_config.yaml")
    try:
        # Use process_all for pipelined processing (recommended)
        transcriber.process_all()

        # Or use process_all_batch for batch processing
        # transcriber.process_all_batch()
    finally:
        transcriber.shutdown()
