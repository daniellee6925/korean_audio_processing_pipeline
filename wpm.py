import os
import wave
import string
import re
import csv
import numpy as np


def clean(text: str) -> str:
    """Clean text by removing punctuation and extra spaces"""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_word(text: str) -> int:
    """Count number of words in a string"""
    text = clean(text)
    return len(text.split())


def count_char(text: str) -> int:
    """Count number of characters in a string"""
    text = clean(text)
    return len(text.replace(" ", ""))


def audio_duration(wav_path: str) -> float:
    """Get duration of a WAV audio file in seconds"""
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration


def compute_from_audio(
    root_dir: str = "archive",
    audio_dir: str = "audio_sentences",
    text_dir: str = "text_sentences",
    csv_output: str = "reading_metrics.csv",
) -> None:
    """compute wpm and cpm from audio with text sentences"""
    results = []

    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        text_path = os.path.join(folder_path, text_dir)
        audio_path = os.path.join(folder_path, audio_dir)

        if not os.path.isdir(text_path) or not os.path.isdir(audio_path):
            continue

        total_words = 0
        total_chars = 0
        total_duration = 0.0

        for txt_file in sorted(os.listdir(text_path)):
            if not txt_file.lower().endswith(".txt"):
                continue

            with open(os.path.join(text_path, txt_file), "r", encoding="utf-8") as f:
                text = f.read()

            total_words += count_word(text)
            total_chars += count_char(text)

            audio_file = txt_file.replace(".txt", ".wav")
            if os.path.exists(os.path.join(audio_path, audio_file)):
                total_duration += audio_duration(os.path.join(audio_path, audio_file))

            total_minutes = total_duration / 60.0
            wpm = total_words / total_minutes if total_minutes > 0 else 0.0
            cpm = total_chars / total_minutes if total_minutes > 0 else 0.0

            results.append(
                {
                    "folder": folder,
                    "senteice_file": txt_file,
                    "total_words": total_words,
                    "total_chars": total_chars,
                    "duration": total_duration,
                    "WPM": wpm,
                    "CPM": cpm,
                }
            )

    # Compute averages
    avg_words = np.mean(np.array([r["total_words"] for r in results]))
    avg_chars = np.mean(np.array([r["total_chars"] for r in results]))
    avg_duration = np.mean(np.array([r["duration"] for r in results]))
    avg_wpm = np.mean(np.array([r["WPM"] for r in results]))
    avg_cpm = np.mean(np.array([r["CPM"] for r in results]))

    with open(os.path.join(root_dir, csv_output), "w", encoding="utf-8") as csvfile:
        fieldnames = ["folder", "senteice_file", "total_words", "total_chars", "duration", "WPM", "CPM"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        # Append average row
        writer.writerow({
            "folder": "AVERAGE",
            "senteice_file": "AVERAGE",
            "total_words": f"{avg_words:.2f}",
            "total_chars": f"{avg_chars:.2f}",
            "duration": f"{avg_duration:.2f}",
            "WPM": f"{avg_wpm:.2f}",
            "CPM": f"{avg_cpm:.2f}"
        })
    print("Saved metrics CSV")


if __name__ == "__main__":
    compute_from_audio()
