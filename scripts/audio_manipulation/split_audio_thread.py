import os
import csv
import wave
import contextlib
import webrtcvad
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple
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

        # log_file = "audio_processor.log"
        # if os.path.exists(log_file):
        #     os.remove(log_file)
        # # Logging
        # logger.add(
        #     "audio_processor.log",
        #     rotation="10 MB",
        #     retention="10 days",
        #     level="INFO",
        # )
        # logger.info("Initialized AudioProcessor")

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
        """Resamples audio to mono, 16-bit, target sample rate and returns new path."""
        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_channels(1).set_frame_rate(self.sample_rate).set_sample_width(2)
        os.makedirs(self.temp_dir, exist_ok=True)
        resampled_path = os.path.join(self.temp_dir, os.path.basename(wav_path))
        audio.export(resampled_path, format=self.file_format)
        # logger.info(f"Resampled to {self.sample_rate} Hz → {resampled_path}")
        return resampled_path

    def split_audio_vad(self, wav_path: str) -> List[Tuple[float, float, float]]:
        """Splits audio into speech segments using WebRTC VAD."""
        # logger.info(f"Splitting file: {wav_path}")
        audio, sr = self.read_wave(wav_path)
        vad = webrtcvad.Vad(self.aggressiveness)

        frame_size = int(sr * self.frame_duration / 1000) * 2
        frames = [audio[i : i + frame_size] for i in range(0, len(audio), frame_size)]

        info = []
        current_time, silence_frames = 0.0, 0
        segment_start = None
        min_silence_frames = self.min_silence_ms // self.frame_duration

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
                    segment_end = current_time - (silence_frames * self.frame_duration / 1000)
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
            current_time += self.frame_duration / 1000

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

    @staticmethod
    def cut_with_ffmpeg(wav_path: str, out_path: str, start_sec: float, end_sec: float):
        """cut audio segments using ffmpeg"""
        command = [
            "ffmpeg",
            "-y",  # overwrite
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-i",
            wav_path,
            "-acodec",
            "copy",  # no re-encoding
            f"{out_path}",
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cut_audio(
        self,
        wav_path: str,
        save_path: str,
        segments: list[tuple[float, float, float]],
    ) -> None:
        """Cuts and exports segments as individual audio files."""

        def process_segment(i, start_sec, end_sec, duration):
            # Decide folder for this segment
            if self.segment_subfolders:
                segment_folder = os.path.join(save_path, f"segment_{i+1}")
                os.makedirs(segment_folder, exist_ok=True)
            else:
                segment_folder = save_path
                os.makedirs(segment_folder, exist_ok=True)

            out_file = os.path.join(segment_folder, f"{self.segment_name}_{i+1}.{self.file_format}")

            self.cut_with_ffmpeg(
                wav_path=wav_path,
                out_path=out_file,
                start_sec=start_sec,
                end_sec=end_sec,
            )

            csv_out_path = os.path.join(segment_folder, f"{self.segment_name}_{i+1}.csv")
            with open(csv_out_path, "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "segment_folder",
                        "segment_file",
                        "start_sec",
                        "end_sec",
                        "duration_sec",
                    ]
                )
                writer.writerow([segment_folder, out_file, start_sec, end_sec, duration])

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, (start, end, dur) in enumerate(segments):
                executor.submit(process_segment, i, start, end, dur)

        # logger.info(f"Exported {len(segments)} segments → {save_path} (threads={self.max_workers})")

    # ------ clean folder ------------
    def clear_temp_files(self):
        """Delete all contents of the temp directory."""
        temp_path = Path(self.temp_dir)
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path)
            temp_path.mkdir()  # recreate empty temp folder
            # logger.info(f"Cleared all temp files in {self.temp_dir}")
        else:
            logger.error(f"No temp folder found at {self.temp_dir}")

    def clear_segment_folders(self):
        """Delete all *_sentences folders under root_dir recursively."""
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
        path = self.resample(wav_path) if self.resample_enabled else wav_path
        segments = self.split_audio_vad(path)
        merged = self.merge_segments(segments)
        self.cut_audio(wav_path=wav_path, save_path=save_path, segments=merged)
        # logger.info(f"Processed file {wav_path}")

    def process_all(self):
        """Run processing on all WAV files in subfolders."""
        os.makedirs(self.output_dir, exist_ok=True)

        # logger.info(f"Scanning {self.root_dir} for audio files...")
        audio_files = []
        for dirpath, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(self.file_format):
                    audio_files.append(os.path.join(dirpath, f))
        # logger.info(f"Found {len(audio_files)} audio files to process")

        def process_one(file_path):
            try:
                rel_path = os.path.relpath(os.path.dirname(file_path), self.root_dir)
                rel_output_dir = os.path.join(self.output_dir, rel_path)
                os.makedirs(rel_output_dir, exist_ok=True)

                file_stem = Path(file_path).stem
                base_save_dir = os.path.join(rel_output_dir, f"{file_stem}_{self.segment_name}")

                if self.segment_subfolders:
                    # If segments get subfolders, just pass the base path;
                    # cut_audio will create numbered segment folders internally
                    save_dir = base_save_dir
                else:
                    # All segments go into a single folder
                    save_dir = base_save_dir
                    os.makedirs(save_dir, exist_ok=True)

                # Pass the actual folder path (save_dir) to process_file
                self.process_file(wav_path=file_path, save_path=save_dir)
                # logger.info(f"Done: {file_path}")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        # Parallel processing
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            futures = {executor.submit(process_one, f): f for f in audio_files}
            for _ in tqdm(
                as_completed(futures),
                total=len(audio_files),
                desc="processing files",
                unit="file",
            ):
                pass
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user.")
        finally:
            executor.shutdown(wait=False)
            logger.info("ThreadPoolExecutor shutdown completed.")

        logger.success(f"Finished processing {len(audio_files)} files total.")
