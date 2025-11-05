from utils import find_files, find_folders, make_dir
import os
import webrtcvad
import contextlib
import wave
import yaml
from loguru import logger
from typing import Tuple, List
from pydub import AudioSegment


# ------ set up logging ------
logger.add(
    "audio_split.log", rotation="10 MB", retention="10 days", level="INFO"
)
logger.info("Starting audio VAD splitting script")


# ------set config-----------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Paths
path_config = config.get("paths", {})
root_dirname = path_config.get("root_dir", "archive")
output_dirname = path_config.get("audio_subdir", "audio_sentences")


# VAD settings
vad_config = config.get("vad", {})
aggressiveness = vad_config.get("aggressiveness", 2)
min_silence_ms = vad_config.get("min_silence_ms", 1000)
sample_rate = vad_config.get("sample_rate", None)
resample = vad_config.get("resample", False)
silence_threshold = vad_config.get("silence_threshold", 0.01)
frame_duration = vad_config.get("frame_duration", 30)


# ------functions-----------
def read_wave(path) -> tuple[bytes, int]:
    """Reads a .wav file and returns PCM audio data and sample rate."""
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "VAD only works on mono audio"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "VAD only works on 16-bit audio"
        sample_rate = wf.getframerate()
        assert sample_rate in (
            8000,
            16000,
            32000,
            48000,
        ), "Invalid sample rate for VAD"
        frames = wf.readframes(wf.getnframes())
        return frames, sample_rate


def resample(
    wav_path: str,
    temp_path: str,
    sample_rate: int = 16000,
    file_format: str = "wav",
) -> None:
    audio = AudioSegment.from_file(wav_path)
    audio = (
        audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
    )
    audio.export(temp_path, format=file_format)


def cut_audio(
    wav_path: str,
    segments: list[tuple[float, float]],
    segment_name: str = "segment_",
    file_format: str = "wav",
):
    audio = AudioSegment.from_file(wav_path)
    for i, (start_sec, end_sec, _) in enumerate(segments):
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        cut_segment = audio[start_ms:end_ms]
        cut_segment.export(
            f"{segment_name}{i+1}.{file_format}", format=file_format
        )


def merge_segments(
    segments: list[tuple[float, float, float]], min_len: float
) -> list[tuple[float, float, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for i in range(1, len(segments)):
        prev_start, prev_end, prev_total = merged[-1]
        start, end, total = segments[i]
        if prev_total < min_len:
            merged[-1] = [prev_start, end, end - prev_start]
        else:
            merged.append(segments[i])
    if merged[-1][2] < min_len:
        start, end, total = merged.pop()
        prev_start, prev_end, prev_total = merged[-1]
        merged[-1] = [prev_start, end, end - prev_start]
    return merged


def split_audio_vad(
    wav_path: str,
    sample_rate: int,
    aggressiveness: int = 2,
    min_silence_ms: float = 1000,  # 1000ms
    frame_duration: float = 30.0,  # 30ms
) -> Tuple[List[bytes], List[Tuple[float, float]]]:
    logger.info(f"Processing file: {wav_path}")

    # try:
    vad = webrtcvad.Vad(aggressiveness)
    audio, sample_rate = read_wave(wav_path)

    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2
    frames = [
        audio[i : i + frame_size] for i in range(0, len(audio), frame_size)
    ]

    segments = []
    info = []
    current_time, silence_duration = 0, 0
    segment = b""
    segment_start = None
    min_silence_frames = min_silence_ms // frame_duration

    for frame in frames:
        if len(frame) < frame_size:
            continue
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            if segment_start is None:
                segment_start = current_time
            segment += frame
        else:
            silence_duration += 1
            if silence_duration >= min_silence_frames:
                segment_end = current_time
                if len(segment) / 2 / sample_rate >= 0.2:
                    segment_total = segment_end - segment_start
                    info.append(
                        (
                            round(segment_start, 3),
                            round(segment_end, 3),
                            round(segment_total, 3),
                        )
                    )
                segment = b""
                segment_start = None
                silence_duration = 0
        current_time += frame_duration / 1000

    if len(segment) > 0:
        if len(segment) / 2 / sample_rate >= 0.2:
            segment_total = segment_end - segment_start
            info.append(
                (
                    round(segment_start, 3),
                    round(segment_end, 3),
                    round(segment_total, 3),
                )
            )
    return info
    # except Exception as e:
    #     logger.error(f"Error processing {wav_path}: {e}")


def main():
    folders = find_folders(root_dirname)[:1]
    logger.info(f"Found folders: {folders}")

    for folder in folders:
        folder_path = f"{root_dirname}/{folder}"
        audio_files = find_files(folder_path, extension=".WAV")
        logger.info(f"Found {len(audio_files)} audio files in {folder_path}")
        for audio_file in audio_files:
            split_audio_vad(
                wav_path=audio_file,
                output_dir=os.path.join(root_dirname, folder, output_dirname),
                sample_rate=sample_rate,
                resample=resample,
                aggressiveness=aggressiveness,
                min_silence_ms=min_silence_ms,
            )


if __name__ == "__main__":
    # main()
    resample(
        wav_path="Tr1.WAV",
        temp_path="temp_folder/temp_16k.wav",
        sample_rate=16000,
        file_format="wav",
    )
    info = split_audio_vad(
        wav_path="temp_folder/temp_16k.wav",
        sample_rate=sample_rate,
        aggressiveness=aggressiveness,
        min_silence_ms=min_silence_ms,
        frame_duration=frame_duration,
    )
    cut_audio(
        wav_path="Tr1.wav",
        segments=info,
        segment_name="sentence_",
        file_format="wav",
    )
    info = merge_segments(info, min_len=10)
    print(info)
