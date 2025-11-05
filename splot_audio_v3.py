import os
import csv
import yaml
import wave
import contextlib
import webrtcvad
import shutil
from loguru import logger
from typing import List, Tuple
from pydub import AudioSegment
from utils import find_files, find_folders
from pathlib import Path


class SplitAudio:
    """Encapsulates audio resampling, VAD segmentation, and cutting."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Paths
        path_config = config.get("paths", {})
        self.root_dir = path_config.get("root_dir", "archive")
        self.output_dir = path_config.get("output_dir", "audio_sentences")
        self.temp_dir = path_config.get("temp_dir", "temp")

        # VAD settings
        vad_config = config.get("vad", {})
        self.aggressiveness = vad_config.get("aggressiveness", 2)
        self.min_silence_ms = vad_config.get("min_silence_ms", 1000)
        self.sample_rate = vad_config.get("sample_rate", 16000)
        self.resample_enabled = vad_config.get("resample", False)
        self.frame_duration = vad_config.get("frame_duration", 30)
        self.file_format = vad_config.get("file_format", "wav")

        # Processing
        processing_config = config.get("processing", {})
        self.min_len = processing_config.get("min_len", 0.0)
        self.segment_name = processing_config.get("segment_name", "segment_")

        log_file = "audio_processor.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        # Logging
        logger.add(
            "audio_processor.log",
            rotation="10 MB",
            retention="10 days",
            level="INFO",
        )
        logger.info("Initialized AudioProcessor")

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
        audio = (
            audio.set_channels(1)
            .set_frame_rate(self.sample_rate)
            .set_sample_width(2)
        )
        os.makedirs(self.temp_dir, exist_ok=True)
        resampled_path = os.path.join(self.temp_dir, os.path.basename(wav_path))
        audio.export(resampled_path, format=self.file_format)
        logger.info(f"Resampled to {self.sample_rate} Hz → {resampled_path}")
        return resampled_path

    def split_audio_vad(
        self, wav_path: str
    ) -> List[Tuple[float, float, float]]:
        """Splits audio into speech segments using WebRTC VAD."""
        logger.info(f"Splitting file: {wav_path}")
        audio, sr = self.read_wave(wav_path)
        vad = webrtcvad.Vad(self.aggressiveness)

        frame_size = int(sr * self.frame_duration / 1000) * 2
        frames = [
            audio[i : i + frame_size] for i in range(0, len(audio), frame_size)
        ]

        info = []
        current_time, silence_frames = 0.0, 0
        segment, segment_start = b"", None
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
                if (
                    silence_frames >= min_silence_frames
                    and segment_start is not None
                ):
                    segment_end = current_time
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
            if duration >= 0.2:
                info.append(
                    (round(segment_start, 3), round(segment_end, 3), duration)
                )

        logger.info(f"Detected {len(info)} speech segments in {wav_path}")
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

        logger.info(
            f"Merged into {len(merged)} segments (min_len={self.min_len}s)"
        )
        return merged

    def cut_audio(
        self,
        wav_path: str,
        save_path: str,
        segments: list[tuple[float, float, float]],
    ) -> None:
        """Cuts and exports segments as individual audio files."""
        audio = AudioSegment.from_file(wav_path)
        output_dir_path = os.path.join(save_path, self.output_dir)
        os.makedirs(output_dir_path, exist_ok=True)

        for i, (start_sec, end_sec, duration) in enumerate(segments):
            cut = audio[int(start_sec * 1000) : int(end_sec * 1000)]
            out_path = os.path.join(
                output_dir_path,
                f"{self.segment_name}{i + 1}.{self.file_format}",
            )
            cut.export(out_path, format=self.file_format)
            csv_out_path = os.path.join(
                output_dir_path,
                f"{self.segment_name}{i + 1}.csv",
            )
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
                writer.writerow(
                    [save_path, out_path, start_sec, end_sec, duration]
                )

        logger.info(f"Exported {len(segments)} segments → {output_dir_path}")

    # ------ clean folder ------------
    def clear_temp_files(self):
        """Delete all contents of the temp directory."""
        temp_path = Path(self.temp_dir)
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path)
            temp_path.mkdir()  # recreate empty temp folder
            logger.info(f"Cleared all temp files in {self.temp_dir}")
        else:
            logger.info(f"No temp folder found at {self.temp_dir}")

    def clear_sentence_folders(self):
        """Delete all *_sentences folders under root_dir recursively."""
        root_path = Path(self.root_dir)
        if not root_path.exists():
            logger.info(f"Root directory {self.root_dir} does not exist")
            return

        count = 0
        for folder in root_path.rglob("*_sentences"):
            if folder.is_dir():
                shutil.rmtree(folder)
                count += 1
        logger.info(
            f"Cleared {count} *_sentences folders under {self.root_dir}"
        )

    def process_file(self, wav_path: str, save_path: str):
        """Full pipeline for one file: resample → split → merge → cut."""
        path = self.resample(wav_path) if self.resample_enabled else wav_path
        segments = self.split_audio_vad(path)
        merged = self.merge_segments(segments)
        self.cut_audio(wav_path=path, save_path=save_path, segments=merged)
        logger.info(f"Processed file {wav_path}")

    def process_all(self):
        """Run processing on all WAV files in subfolders."""
        folders = find_folders(self.root_dir)
        logger.info(f"Found {len(folders)} folders in {self.root_dir}")

        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)
            audio_files = find_files(
                folder_path, extension=f".{self.file_format}"
            )
            logger.info(f"Processing {len(audio_files)} files in {folder_path}")
            for file in audio_files:
                try:
                    file_stem = os.path.splitext(os.path.basename(file))[0]
                    save_path = os.path.join(
                        folder_path, file_stem + "_sentences"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    self.process_file(wav_path=file, save_path=save_path)
                except Exception as e:
                    logger.error(f"Failed to process {file}: {e}")
