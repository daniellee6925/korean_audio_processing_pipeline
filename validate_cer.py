import os
import whisper
import torch
from pathlib import Path
from jiwer import wer
import csv


device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["WHISPER_CACHE_DIR"] = "/fsx/models"

model = whisper.load_model("large-v3", device=device)


def validate_sentence_split(
    root_dir: str = "/workspace/archive",
    audio_dir: str = "audio_sentences",
    text_dir: str = "text_sentences",
    wer_threshold: float = 0.25,
    csv_output: str = "validate_sentence_split.csv",
) -> None:
    """compare audio sentence splits with text sentences ausing WER metric"""
    with open(os.path.join(root_dir, csv_output), "w", encoding="utf-8") as csvfile:
        fieldnames = ["folder", "num_sentences", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for folder in sorted(os.listdir(root_dir))[:1]:
            folder_path = os.path.join(root_dir, folder)
            text_path = os.path.join(folder_path, text_dir)
            audio_path = os.path.join(folder_path, audio_dir)

            if not os.path.exists(text_path) or not os.path.exists(audio_path):
                continue

            text_files = sorted(Path(text_path).glob("*.txt"))
            audio_files = sorted(Path(audio_path).glob("*.wav"))

            if len(audio_files) != len(text_files):
                print(
                    f"Skipping {folder_path}, mismatch in number of audio and text files"
                )
                writer.writerow(
                    {
                        "folder": folder,
                        "num_sentences": len(audio_files),
                        "status": "FAIL",
                    }
                )
                continue

            folder_status = "PASS"
            for audio_file, text_file in zip(audio_files, text_files):
                # Transcribe audio file
                result = model.transcribe(str(audio_file))
                predicted_text = result["text"].strip()

                # Read reference text
                with open(text_file, "r", encoding="utf-8") as f:
                    reference_text = f.read().strip()

                # Compute WER
                error = wer(reference_text, predicted_text)
                if error > wer_threshold:
                    folder_status = "FAIL"
                    break
                print(reference_text)
                print(predicted_text)
                print(f"error: {error}")

            writer.writerow(
                {
                    "folder": folder,
                    "num_sentences": len(audio_files),
                    "status": folder_status,
                }
            )

            print(f"Validated folder: {folder_path}")
        print(f"Validation results saved to {csv_output}")


if __name__ == "__main__":
    validate_sentence_split()
