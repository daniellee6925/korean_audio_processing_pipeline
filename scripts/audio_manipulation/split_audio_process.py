import os
import csv
import yaml
import wave
import contextlib
import webrtcvad
import shutil
import ffmpeg
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple, Dict, Any
from pydub import AudioSegment
from pathlib import Path


class SplitAudio:
    """Encapsulates audio resampling, VAD segmentation, and cutting."""

    def __init__(
        self,
        # Path parameters
        root_dir: str = "archive",
        output_dir: str = "audio_sentences",
        temp_dir: str = "temp",
        # VAD parameters
        aggressiveness: int = 2,
        min_silence_ms: int = 1000,
        min_segment_ms: float = 200.0,
        sample_rate: int = 16000,
        resample_enabled: bool = True,
        frame_duration: int = 30,
        file_format: str = "wav",
        # Processing parameters
        min_len: float = 0.0,
        segment_name: str = "segment",
        max_workers: int = 8,
        segment_subfolders: bool = False,
        batch_size: int = 10,  # segments per FFmpeg call
    ):
        # Paths
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir

        # VAD settings
        self.aggressiveness = aggressiveness
        self.min_silence_ms = min_silence_ms
        self.min_segment_ms = min_segment_ms
        self.sample_rate = sample_rate
        self.resample_enabled = resample_enabled
        self.frame_duration = frame_duration
        self.file_format = file_format

        # Processing
        self.min_len = min_len
        self.segment_name = segment_name
        self.max_workers = max_workers
        self.segment_subfolders = segment_subfolders
        self.batch_size = batch_size

        # Cache frame size calculation
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000) * 2
        self.min_silence_frames = self.min_silence_ms // self.frame_duration

        # Logging setup (optional)
        # log_file = "audio_processor.log"
        # if os.path.exists(log_file):
        #     os.remove(log_file)
        # logger.add(
        #     log_file,
        #     rotation="10 MB",
        #     retention="10 days",
        #     level="INFO",
        # )
        # logger.info(f"Initialized AudioProcessor with {self.max_workers} workers")

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def read_wave(self, path: str) -> tuple[bytes, int]:
        """Reads an audio file and returns PCM audio data and sample rate."""
        with contextlib.closing(wave.open(path, "rb")) as wf:
            assert wf.getnchannels() == 1, "VAD only works on mono audio"
            assert wf.getsampwidth() == 2, "VAD only works on 16-bit audio"
            sr = wf.getframerate()
            assert sr in (
                8000,
                16000,
                32000,
                48000,
            ), f"Invalid sample rate: {sr}"
            frames = wf.readframes(wf.getnframes())
            return frames, sr

    def resample(self, wav_path: str) -> str:
        """Resamples audio to mono, 16-bit, target sample rate using ffmpeg-python."""
        os.makedirs(self.temp_dir, exist_ok=True)
        resampled_path = os.path.join(self.temp_dir, os.path.basename(wav_path))

        try:
            # Use ffmpeg-python for faster resampling
            stream = ffmpeg.input(wav_path)
            stream = ffmpeg.output(
                stream,
                resampled_path,
                acodec="pcm_s16le",
                ac=1,  # mono
                ar=self.sample_rate,
                loglevel="error",
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            # logger.info(f"Resampled to {self.sample_rate} Hz → {resampled_path}")
            return resampled_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg resampling error: {e.stderr.decode()}")
            # Fallback to pydub if ffmpeg fails
            audio = AudioSegment.from_file(wav_path)
            audio = audio.set_channels(1).set_frame_rate(self.sample_rate).set_sample_width(2)
            audio.export(resampled_path, format=self.file_format)
            return resampled_path

    def split_audio_vad(self, wav_path: str) -> List[Tuple[float, float, float]]:
        """Splits audio into speech segments using WebRTC VAD."""
        # logger.info(f"Splitting file: {wav_path}")
        audio, sr = self.read_wave(wav_path)
        vad = webrtcvad.Vad(self.aggressiveness)

        frame_size = self.frame_size
        frames = [audio[i : i + frame_size] for i in range(0, len(audio), frame_size)]

        info = []
        current_time = 0.0
        silence_frames = 0
        segment_start = None
        min_silence_frames = self.min_silence_frames
        frame_duration_sec = self.frame_duration / 1000

        for frame in frames:
            if len(frame) < frame_size:
                continue
            if vad.is_speech(frame, sr):
                if segment_start is None:
                    segment_start = current_time
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= min_silence_frames and segment_start is not None:
                    segment_end = current_time - (silence_frames * frame_duration_sec)
                    duration = round(segment_end - segment_start, 3)
                    if duration >= 0.2:
                        info.append(
                            (
                                round(segment_start, 3),
                                round(segment_end, 3),
                                duration,
                            )
                        )
                    segment_start = None
            current_time += frame_duration_sec

        # Handle trailing segment
        if segment_start is not None:
            segment_end = current_time
            duration = round(segment_end - segment_start, 3)
            if duration >= self.min_segment_ms / 1000:
                info.append((round(segment_start, 3), round(segment_end, 3), duration))

        # logger.info(f"Detected {len(info)} speech segments in {wav_path}")
        return info

    def merge_segments(
        self, segments: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Merge short segments under min_len with the next one."""
        if not segments:
            return []

        merged = [segments[0]]
        for i in range(1, len(segments)):
            prev_start, prev_end, prev_total = merged[-1]
            start, end, total = segments[i]
            if prev_total < self.min_len:
                merged[-1] = (
                    round(prev_start, 3),
                    round(end, 3),
                    round(end - prev_start, 3),
                )
            else:
                merged.append(segments[i])

        # Merge last if still too short
        if len(merged) > 1 and merged[-1][2] < self.min_len:
            last_start, last_end, _ = merged.pop()
            prev_start, prev_end, _ = merged[-1]
            merged[-1] = (
                round(prev_start, 3),
                round(last_end, 3),
                round(last_end - prev_start, 3),
            )

        # logger.info(f"Merged into {len(merged)} segments (min_len={self.min_len}s)")
        return merged

    def cut_segments(
        self, wav_path: str, segments_to_cut: List[Tuple[int, str, float, float]]
    ) -> None:
        """Cut multiple segments sequentially, preserving original quality."""
        if not segments_to_cut:
            return

        for seg_idx, out_file, start, end in segments_to_cut:
            try:
                stream = ffmpeg.input(wav_path, ss=start, to=end)
                stream = ffmpeg.output(
                    stream,
                    out_file,
                    acodec="copy",  # Preserve original codec/quality
                    loglevel="error",
                )
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg cut error for {out_file}: {e.stderr.decode()}")

    def cut_audio(
        self,
        wav_path: str,
        save_path: str,
        segments: list[tuple[float, float, float]],
    ) -> None:
        """Cuts and exports segments using batched FFmpeg calls."""
        segment_data = []
        segments_to_cut = []

        for i, (start_sec, end_sec, duration) in enumerate(segments):
            if self.segment_subfolders:
                segment_folder = os.path.join(save_path, f"segment_{i+1}")
                os.makedirs(segment_folder, exist_ok=True)
            else:
                segment_folder = save_path

            out_file = os.path.join(segment_folder, f"{self.segment_name}_{i+1}.{self.file_format}")

            # Add to batch
            segments_to_cut.append((i, out_file, start_sec, end_sec))
            segment_data.append([segment_folder, out_file, start_sec, end_sec, duration])

        if segments_to_cut:
            self.cut_segments(wav_path, segments_to_cut)

        # Write consolidated CSV
        csv_path = os.path.join(save_path, f"{self.segment_name}_all.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["segment_folder", "segment_file", "start_sec", "end_sec", "duration_sec"]
            )
            writer.writerows(segment_data)

        # logger.info(f"Exported {len(segments)} segments → {save_path}")

    def clear_temp_files(self):
        """Delete all contents of the temp directory."""
        temp_path = Path(self.temp_dir)
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path)
            temp_path.mkdir()
            # logger.info(f"Cleared all temp files in {self.temp_dir}")
        else:
            logger.error(f"No temp folder found at {self.temp_dir}")

    def clear_segment_folders(self):
        """Delete all *_segment folders under root_dir recursively."""
        root_path = Path(self.root_dir)
        if not root_path.exists():
            # logger.info(f"Root directory {self.root_dir} does not exist")
            return

        count = 0
        for folder in root_path.rglob("*_sentences"):
            if folder.is_dir():
                shutil.rmtree(folder)
                count += 1
        # logger.info(f"Cleared {count} *_sentences folders under {self.root_dir}")

    def process_file(self, wav_path: str, save_path: str):
        """Full pipeline for one file: resample → split → merge → cut."""
        # Resample ONLY for VAD detection 16kHz
        vad_path = self.resample(wav_path)
        segments = self.split_audio_vad(vad_path)
        merged = self.merge_segments(segments)

        # Cut from ORIGINAL file to preserve quality and sample rate
        self.cut_audio(wav_path=wav_path, save_path=save_path, segments=merged)
        # logger.info(f"Processed file {wav_path}")

    def process_all(self):
        """Run processing on all WAV files using ProcessPoolExecutor."""
        os.makedirs(self.output_dir, exist_ok=True)

        # logger.info(f"Scanning {self.root_dir} for audio files...")
        audio_files = []
        for dirpath, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(self.file_format):
                    audio_files.append(os.path.join(dirpath, f))
        # logger.info(f"Found {len(audio_files)} audio files to process")

        # Prepare arguments for parallel processing
        process_args = []
        for file_path in audio_files:
            rel_path = os.path.relpath(os.path.dirname(file_path), self.root_dir)
            rel_output_dir = os.path.join(self.output_dir, rel_path)
            file_stem = Path(file_path).stem
            base_save_dir = os.path.join(rel_output_dir, f"{file_stem}_{self.segment_name}")

            process_args.append((file_path, base_save_dir))

        # Use ProcessPoolExecutor for CPU-bound VAD work
        executor = ProcessPoolExecutor(max_workers=self.max_workers)
        try:
            # Submit all tasks
            future_to_args = {
                executor.submit(self._process_wrapper, args): args for args in process_args
            }

            # Process completed tasks with progress bar
            success_count = 0
            with tqdm(total=len(process_args), desc="Spliting Aduio files", unit="file") as pbar:
                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                    except Exception as e:
                        args = future_to_args[future]
                        logger.error(f"Task failed for {args[0]}: {e}")
                    finally:
                        pbar.update(1)
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user.")
        finally:
            executor.shutdown(wait=False)
            logger.info("ThreadPoolExecutor shutdown completed.")

        logger.success(
            f"Finished: {success_count}/{len(audio_files)} files processed successfully."
        )

    def _process_wrapper(self, args):
        """Wrapper function for parallel processing."""
        file_path, base_save_dir = args
        try:
            os.makedirs(base_save_dir, exist_ok=True)
            self.process_file(wav_path=file_path, save_path=base_save_dir)
            # logger.info(f"Done: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return False
