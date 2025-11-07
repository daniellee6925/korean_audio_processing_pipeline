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
    ratio_threshold: float = 0.25,
    csv_output: str = "validate_sentence_split.csv",
) -> None:
    """Validate audio/text sentence splits by approximate word count ratio."""

    with open(
        os.path.join(root_dir, csv_output), "w", encoding="utf-8"
    ) as csvfile:
        fieldnames = ["folder", "num_sentences", "avg ratio diff", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for folder in sorted(os.listdir(root_dir)):
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
                        "avg ratio diff": "Sentence count mismatch",
                        "status": "FAIL",
                    }
                )
                continue

            ratio_diffs = []
            folder_status = "PASS"
            for audio_file, text_file in zip(audio_files, text_files):
                # Transcribe audio file
                result = model.transcribe(str(audio_file))
                predicted_text = result["text"].strip()

                # Read reference text
                with open(text_file, "r", encoding="utf-8") as f:
                    reference_text = f.read().strip()

                ref_count = len(reference_text.split())
                pred_count = len(predicted_text.split())

                if ref_count == 0:
                    continue

                ratio_diff = abs(pred_count - ref_count) / ref_count
                ratio_diffs.append(ratio_diff)
                if ratio_diff > ratio_threshold:
                    folder_status = "FAIL"

            avg_ratio_diff = (
                sum(ratio_diffs) / len(ratio_diffs) if ratio_diffs else 0.0
            )
            writer.writerow(
                {
                    "folder": folder,
                    "num_sentences": len(audio_files),
                    "avg ratio diff": f"{avg_ratio_diff:.4f}",
                    "status": folder_status,
                }
            )

            print(f"Validated folder: {folder_path}")
        print(f"Validation results saved to {csv_output}")


if __name__ == "__main__":
    validate_sentence_split()
