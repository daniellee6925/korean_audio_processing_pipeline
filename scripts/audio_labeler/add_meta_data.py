import os
import json
import numpy as np
import soundfile as sf
import pyworld as pw
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm

ROOT_DIR = "Voice_Bank/voice_pick"
GAP_THRESHOLD = 1.0  # seconds
N_SAMPLES_PITCH = 10  # reduce from 20 for speed
DOWNSAMPLE_SR = 16000  # pitch calculation


# ------------------- SPEED -------------------
def vectorized_speed(transcription, duration, gap_threshold=GAP_THRESHOLD):
    words_raw = transcription.strip().split()
    n_words = len(words_raw)
    if n_words == 0 or duration <= 0:
        return None, None
    starts = np.arange(n_words) * duration / n_words
    ends = (np.arange(n_words) + 1) * duration / n_words
    gaps = starts[1:] - ends[:-1]
    long_gaps = np.sum(gaps[gaps > gap_threshold])
    effective_duration = duration - long_gaps
    speed_val = n_words / effective_duration if effective_duration > 0 else None
    # label
    if speed_val is None:
        speed_label = None
    elif speed_val < 1.0:
        speed_label = "Slow"
    elif speed_val < 1.3:
        speed_label = "Slow-Medium"
    elif speed_val < 1.6:
        speed_label = "Medium"
    elif speed_val < 2.0:
        speed_label = "Medium-Fast"
    else:
        speed_label = "Fast"
    return speed_val, speed_label


# ------------------- PITCH -------------------
def fast_pitch_sampled(
    fpath, n_samples=N_SAMPLES_PITCH, fmin=50, fmax=400, max_retries=3, default_pitch=150.0
):
    for attempt in range(max_retries):
        try:
            y, sr = sf.read(fpath)
            if y.ndim > 1:
                y = y.mean(axis=1)
            # downsample if necessary
            if sr > DOWNSAMPLE_SR:
                factor = sr // DOWNSAMPLE_SR
                y = y[::factor]
                sr = DOWNSAMPLE_SR
            total_len = len(y)
            segment_len = int(0.2 * sr)
            if total_len < segment_len:
                return default_pitch
            step = max(total_len // n_samples, segment_len)
            pitches = []
            for start in range(0, total_len - segment_len, step):
                seg = y[start : start + segment_len]
                _f0, t = pw.harvest(seg.astype(np.float64), sr, f0_floor=fmin, f0_ceil=fmax)
                f0 = pw.stonemask(seg.astype(np.float64), _f0, t, sr)
                f0 = f0[f0 > 0]
                if len(f0) > 0:
                    pitches.append(np.median(f0))
            if len(pitches) > 0:
                return float(np.median(pitches))
            return default_pitch
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"[retry {attempt+1}] {fpath} ({e})")
            else:
                return default_pitch


def pitch_to_label(pitch, low_thr=120, high_thr=180):
    if pitch < low_thr:
        return "Low"
    elif pitch < high_thr:
        return "Mid"
    else:
        return "High"


# ------------------- AGE CLASSIFICATION -------------------
def classify_age(age_str):
    if not age_str:
        return None
    age_str = age_str.lower()
    if "child" in age_str or "10" in age_str:
        return "child"
    elif "20" in age_str or "30" in age_str or "young" in age_str:
        return "young"
    elif "40" in age_str or "50" in age_str or "middle" in age_str:
        return "middle"
    else:
        return "old"


# ------------------- PROCESS SINGLE JSON -------------------
def process_json_audio(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        audio_path = data.get("filepath")
        if not audio_path or not os.path.exists(audio_path):
            logger.debug(f"[SKIP] Audio file not found for {json_path}")
            return json_path

        transcription = data.get("transcription", "")
        audio_length = data.get("audio_length", 0.0)

        # --- Compute speed ---
        speed_val, speed_label = vectorized_speed(transcription, audio_length)
        data["speed"] = speed_label

        # --- Compute pitch ---
        pitch_val = fast_pitch_sampled(audio_path)
        pitch_label = pitch_to_label(pitch_val)
        data["pitch"] = pitch_label

        # --- Flatten metadata ---
        metadata = data.get("metadata", {})
        data["gender"] = metadata.get("gender", None)
        data["age"] = classify_age(metadata.get("age", None))
        data["style"] = metadata.get("traits", None)
        if "metadata" in data:
            del data["metadata"]

        # --- Save updated JSON ---
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # logger.info(f"Processed: {json_path}")
        return json_path
    except Exception as e:
        logger.error(f"[ERROR] {json_path} -> {e}")
        return None


# ------------------- MAIN FUNCTION -------------------
def process_root_dir_parallel(root_dir, max_workers=4):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".json"):
                json_files.append(os.path.join(dirpath, file))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_json_audio, f) for f in json_files]

        # Wrap as_completed with tqdm for progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON files"):
            future.result()  # trigger exceptions if any
    logger.success(f"Finished Processing {len(json_files)} files")


if __name__ == "__main__":
    process_root_dir_parallel(ROOT_DIR, max_workers=8)  # adjust workers to CPU cores
