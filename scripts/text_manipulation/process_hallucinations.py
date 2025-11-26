import json
from pathlib import Path
from difflib import SequenceMatcher
from loguru import logger
import hgtk
import re
import shutil
import pandas as pd
import numpy as np


def hangul_to_jamo(word):
    try:
        return "".join(hgtk.text.decompose(word).split())
    except Exception:
        return word


def phonetic_similarity(w1, w2):
    return SequenceMatcher(None, hangul_to_jamo(w1), hangul_to_jamo(w2)).ratio()


def phonetic_sentence_similarity(s1, s2):
    words1 = s1.split()
    words2 = s2.split()

    if not words1 or not words2:
        return 0.0

    scores = []

    # align words by index (simple but effective)
    for w1, w2 in zip(words1, words2):
        sim = phonetic_similarity(w1, w2)
        scores.append(sim)

    # if different lengths, penalize missing words
    length_penalty = min(len(words1), len(words2)) / max(len(words1), len(words2))

    if scores:
        return np.mean(scores) * length_penalty
    return 0.0


def process_csv(input_path: str):
    input_path = Path(input_path)

    file_name = input_path.name
    folder_name = input_path.parent.name
    file_stem = input_path.stem

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    filtered_df = df[df["flagged"] == "FLAG"]

    output_dir = Path("output") / folder_name / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in filtered_df.itertuples(index=False):
        transcribed = row.transcribed
        if pd.isna(row.transcribed) or re.search(r"\d", transcribed):
            continue
        json_entry = {
            "foldername": folder_name,
            "filename": file_name,
            "segment_idx": row.segment_idx,
            "time range": [row.start_sec, row.end_sec],
            "ground truth": row.text,
            "transcribed": row.transcribed,
            "hallucination": categorize_hallucination(row.text, row.transcribed),
        }
        output_path = output_dir / f"{row.segment_idx}.json"
        with open(output_path, "w", encoding="utf-8-sig") as f_out:
            json.dump(json_entry, f_out, ensure_ascii=False, indent=4)
    logger.info(f"finished processing {input_path}")


def categorize_hallucination(text, transcribed, phonetic_threshold=0.7):
    """categorizes hallucinations into inaudible, homophone, and semantic"""
    INAUDIBLE_TOKENS = {"음", "[LAUGHTER]", "[SONANT]", "[ENS]", "[*]"}

    if any(tok in text for tok in INAUDIBLE_TOKENS) and transcribed:
        return "Halllucination from Inaudible"
    elif phonetic_sentence_similarity(text, transcribed) > phonetic_threshold:
        return "Homophone"
    else:
        return "Semantic"


def get_audio(audio_path: str, json_path):
    json_path = Path(json_path)
    audio_path = Path(audio_path)

    json_files = list(json_path.rglob("*.json"))
    copied = 0
    missing = 0
    for json_file in json_files:
        stem = json_file.stem
        rel_path = json_file.parent.relative_to(json_path)
        candidate = audio_path / rel_path / f"{stem}.wav"
        if candidate.exists():
            shutil.copy2(candidate, json_path / rel_path / candidate.name)
            copied += 1
        else:
            missing += 1
    logger.info(
        f"Finished copying {copied} files from {audio_path} -> {json_path}. Missing {missing} files"
    )


def process_all(root_dir: str):
    root_path = Path(root_dir)
    csv_files = list(root_path.glob("*.csv"))
    for csv_file in csv_files:
        process_csv(csv_file)

    logger.success(f"Completed processing {len(csv_files)} csv files")


if __name__ == "__main__":
    # process_all("Korean_Conversational_Speech_Corpus")
    get_audio(audio_path="Results", json_path="Korean_Conversational_Speech_Corpus")
