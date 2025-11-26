from split_audio.split_audio_thread import SplitAudio
from pathlib import Path
import csv


def main(wav_path: str, segment_path: str, out_path: str):
    processor = SplitAudio()

    segment_path = Path(segment_path)
    wav_path = Path(wav_path)
    out_path = Path(out_path)
    csv_files = list(segment_path.glob("*.csv"))

    for csv_file in csv_files:

        base = csv_file.stem
        audio_path = wav_path / f"{base}.wav"
        segments = []
        with csv_file.open("r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                start, end = float(row[0]), float(row[1])
                segments.append((start, end, end - start))

        save_path = out_path / base

        processor.cut_audio(wav_path=audio_path, save_path=save_path, segments=segments)


if __name__ == "__main__":
    main(wav_path="data/WAV", segment_path="data/extracted_folder", out_path="Results")
