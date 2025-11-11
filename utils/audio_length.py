from pathlib import Path
import soundfile as sf
from tqdm import tqdm


def get_total_wav_length(root_dir: str) -> None:
    root_path = Path(root_dir)
    wav_files = list(root_path.rglob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in: {root_dir}")
        return

    total_seconds = 0.0

    for wav_path in tqdm(wav_files, desc="Calculating total duration"):
        try:
            with sf.SoundFile(wav_path) as audio:
                total_seconds += len(audio) / audio.samplerate
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")

    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(
        f"\nTotal duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s ({total_seconds:.2f} sec total)"
    )
    print(f"Total files processed: {len(wav_files)}")


if __name__ == "__main__":
    get_total_wav_length("1107_Recording")
