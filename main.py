from split_audio.split_audio_long_config import SplitAudio
import os
import csv
from pathlib import Path
import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} runtime: {end - start:.2f} seconds")
        return result

    return wrapper


@timer
def main():
    processor = SplitAudio(config_path="configs/split_config.yaml")
    processor.clear_segment_folders()
    processor.process_all()
    # processor.process_file(
    #     wav_path="sentence_14/160101_014_Tr1.wav", save_path="sentence_14"
    # )
    processor.clear_temp_files()


def read_segments_from_csv(csv_path):
    segments = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
                duration = end - start
                if duration > 0:
                    segments.append((start, end, duration))
            except (ValueError, KeyError):
                continue
    return segments


def split():
    dir_path = Path("Korean_Conversational_Speech_Corpus")
    processor = SplitAudio(config_path="config.yaml")
    for wav_file in dir_path.glob("*.wav"):
        csv_file = dir_path / f"{wav_file.stem}.csv"
        if not csv_file.exists():
            print(f"No CSV for {wav_file}, skipping...")
            continue
        segments = read_segments_from_csv(csv_file)

        if not segments:
            print(f"No valid segments in {csv_file}, skipping...")
            continue

        save_path = dir_path / f"{wav_file.stem}_segments"
        save_path.mkdir(exist_ok=True)

        processor.cut_audio(wav_path=str(wav_file), save_path=str(save_path), segments=segments)
        print(f"Processed {wav_file} → {len(segments)} segments")


if __name__ == "__main__":
    main()
    # split()
