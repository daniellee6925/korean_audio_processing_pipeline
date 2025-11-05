import os
import wave
import webrtcvad
import contextlib
import numpy as np
import csv
from typing import Tuple
from pydub import AudioSegment


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


def write_wave(path, audio, sample_rate) -> None:
    """Writes PCM audio data to a .wav file."""
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def measure_and_trim_silence(
    segment: bytes, sample_rate: int, threshold: float = 0.05, window_ms: int = 20
) -> Tuple[bytes, float, float]:
    """
    Trim leading and trailing silence from a segment.
    Returns:
        trimmed_segment: bytes
        leading_ms: float
        trailing_ms: float
    """
    samples = np.frombuffer(segment, dtype=np.int16).astype(np.float32)
    window_size = int(sample_rate * window_ms / 1000)

    if len(samples) < window_size:
        return segment

    energy = np.array(
        [np.sqrt(np.mean(samples[i : i + window_size] ** 2)) for i in range(0, len(samples), window_size)]
    )

    energy /= np.max(energy) if np.max(energy) > 0 else 1.0

    # keep only windows with energy above threshold)
    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        return segment, 0.0, 0.0

    start_idx = max(0, voiced[0] * window_size)
    end_idx = min((voiced[-1] + 1) * window_size, len(samples))

    leading_ms = voiced[0] * window_ms
    trailing_ms = (len(energy) - voiced[-1] - 1) * window_ms

    trimmed = samples[start_idx:end_idx].astype(np.int16).tobytes()

    trimmed_samples = np.atleast_1d(trimmed)
    non_zero = np.nonzero(trimmed_samples)[0]

    if len(non_zero) != 0:
        leading_ms_pst_trim = trailing_ms_post_trim = 0.0
    else:
        first_non_zero = non_zero[0]
        last_non_zero = non_zero[-1]
        leading_ms_pst_trim = first_non_zero / sample_rate * 1000
        trailing_ms_post_trim = (len(trimmed) - last_non_zero - 1) / sample_rate * 1000

    return trimmed, leading_ms, trailing_ms, leading_ms_pst_trim, trailing_ms_post_trim


def vad_split(wav_path, aggressiveness=2, min_silence_ms=1000) -> tuple[list[bytes], int, list[dict]]:
    """Splits a mono, 16-bit, 16kHz WAV file using WebRTC VAD."""

    vad = webrtcvad.Vad(aggressiveness)
    audio, sample_rate = read_wave(wav_path)

    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2
    frames = [audio[i : i + frame_size] for i in range(0, len(audio), frame_size)]

    segments = []
    info = []
    segment = b""
    min_silence_frames = min_silence_ms // frame_duration
    for frame in frames:
        if len(frame) < frame_size:
            continue
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            silence_duration = 0
            segment += frame
        else:
            silence_duration += 1
            if silence_duration >= min_silence_frames:
                if len(segment) > 0:
                    # trim silence from start and end
                    trimmed_segment, lead_ms, trail_ms, lead_ms_post, trail_ms_post = measure_and_trim_silence(
                        segment, sample_rate
                    )
                    if len(segment) / 2 / sample_rate >= 0.2:
                        segments.append(trimmed_segment)
                        info.append(
                            {
                                "total_ms_post_trim": len(trimmed_segment) / 2 / sample_rate * 1000,
                                "leading_ms_pre_trim": lead_ms,
                                "trailing_ms_pre_trim": trail_ms,
                                "leading_ms_post_trim": lead_ms_post,
                                "trailing_ms_post_trim": trail_ms_post,
                            }
                        )
                    segment = b""
                silence_duration = 0

    if len(segment) > 0:
        segments.append(segment)
        trimmed_segment, lead_ms, trail_ms, lead_ms_post, trail_ms_post = measure_and_trim_silence(segment, sample_rate)
        if len(segment) / 2 / sample_rate >= 0.2:
            segments.append(trimmed_segment)
            info.append(
                {
                    "total_ms_post_trim": len(trimmed_segment) / 2 / sample_rate * 1000,
                    "leading_ms_pre_trim": lead_ms,
                    "trailing_ms_pre_trim": trail_ms,
                    "leading_ms_post_trim": lead_ms_post,
                    "trailing_ms_post_trim": trail_ms_post,
                }
            )
    return segments, sample_rate, info


def main(root_dir="archive") -> None:
    for idx, folder in enumerate(sorted(os.listdir(root_dir))):
        folder_path = os.path.join(root_dir, folder)
        wav_path = os.path.join(folder_path, "Tr1.WAV")

        if not os.path.isfile(wav_path):
            continue

        print(f"Processing {wav_path}...")

        # Convert to mono, 16-bit, 16kHz
        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        temp_path = os.path.join(folder_path, "temp_16k.wav")
        audio.export(temp_path, format="wav")

        # Run VAD
        segments, sample_rate, info = vad_split(temp_path)
        with open(os.path.join(folder_path, f"split_info_{idx+2:003}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "total_ms_post_trim",
                    "leading_ms_pre_trim",
                    "trailing_ms_pre_trim",
                    "leading_ms_post_trim",
                    "trailing_ms_post_trim",
                ],
            )
            writer.writeheader()
            writer.writerows(info)

        output_dir = os.path.join(folder_path, "audio_sentences")
        os.makedirs(output_dir, exist_ok=True)

        for i, seg in enumerate(segments):
            out_file = os.path.join(output_dir, f"sentence_{i+1}.wav")
            write_wave(out_file, seg, sample_rate)
            print(f"Saved {out_file}")

        os.remove(temp_path)


if __name__ == "__main__":
    main()
