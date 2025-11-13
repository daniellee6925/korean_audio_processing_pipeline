import os
import json
import gradio as gr
import tempfile
import base64

# --------------------
# Config
# --------------------
AUDIO_DIR = "audio_files"
OUTPUT_JSON = "metadata.json"
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac")
CSV_EXT = ".csv"

# --------------------
# Collect audio files recursively
# --------------------
audio_files = []
for root, _, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.lower().endswith(ALLOWED_EXTENSIONS):
            rel_path = os.path.relpath(os.path.join(root, f), AUDIO_DIR)
            audio_files.append(rel_path)

total_files = len(audio_files)

# --------------------
# Initialize JSON
# --------------------
transcriptions = {}
start_idx = 0
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        transcriptions = json.load(f)
    # Determine last completed audio index
    completed_files = list(transcriptions.keys())
    if completed_files:
        last_file = completed_files[-1]
        if last_file in audio_files:
            start_idx = audio_files.index(last_file) + 1


# --------------------
# Helper functions
# --------------------
def encode_audio(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_segment_info(file_rel_path):
    """Load CSV info if exists"""
    csv_file = os.path.splitext(file_rel_path)[0] + ".csv"
    csv_path = os.path.join(AUDIO_DIR, csv_file)

    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        segment_folder, segment_file, start_sec, end_sec, duration_sec = parts
        try:
            start_sec = float(start_sec)
            end_sec = float(end_sec)
            duration_sec = float(duration_sec)
        except ValueError:
            continue
        return {
            "segment_folder": segment_folder,
            "segment_file": segment_file,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
        }
    return None


def get_preload_transcript(file_rel_path):
    """Load suggested transcript from .txt in same folder"""
    folder_path = os.path.dirname(os.path.join(AUDIO_DIR, file_rel_path))
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not txt_files:
        return ""
    txt_file_path = os.path.join(folder_path, txt_files[0])
    audio_file_name = os.path.basename(file_rel_path)

    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if audio_file_name in line:
            return line.strip()
    return lines[0].strip() if lines else ""


def save_transcription(file_rel_path, transcription):
    segment_info = get_segment_info(file_rel_path)
    if segment_info is None:
        tar_file = os.path.basename(os.path.dirname(file_rel_path))
        wav_file = os.path.basename(file_rel_path)
        time_range = None
    else:
        tar_file = segment_info["segment_folder"]
        wav_file = segment_info["segment_file"]
        time_range = [segment_info["start_sec"], segment_info["end_sec"]]

    audio_full_path = os.path.join(AUDIO_DIR, file_rel_path)
    audio_content = encode_audio(audio_full_path)

    transcriptions[file_rel_path.replace("\\", "/")] = {
        "audio_segment": audio_content,
        "transcript": transcription,
        "tar_file": tar_file,
        "wav_file": wav_file,
        "time_range": time_range,
    }

    # atomic write
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmpf:
        json.dump(transcriptions, tmpf, indent=4, ensure_ascii=False)
        tmp_name = tmpf.name
    os.replace(tmp_name, OUTPUT_JSON)


def get_current_audio(idx):
    if 0 <= idx < total_files:
        rel_path = audio_files[idx]
        audio_path = os.path.join(AUDIO_DIR, rel_path)
        preloaded_transcript = get_preload_transcript(rel_path)
        return audio_path, preloaded_transcript
    return None, ""


def next_audio(idx, transcription):
    if idx >= total_files:
        return None, "✅ All files transcribed.", idx, ""

    current_rel_path = audio_files[idx]
    if transcription.strip() == "":
        transcription = "[INAUDIBLE]"

    save_transcription(current_rel_path, transcription)

    new_idx = idx + 1
    if new_idx >= total_files:
        return None, f"✅ All {total_files} files transcribed.", new_idx, ""
    else:
        next_file = audio_files[new_idx]
        status = f"{new_idx+1}/{total_files} - Now transcribing {os.path.basename(next_file)}"
        audio_path, preloaded_transcript = get_current_audio(new_idx)
        return audio_path, status, new_idx, preloaded_transcript


def skip_audio(idx):
    return next_audio(idx, "[INAUDIBLE]")


# --------------------
# Gradio UI
# --------------------
first_audio, first_transcript = get_current_audio(start_idx)

with gr.Blocks(title="Audio Transcription Tool", theme="soft") as demo:
    gr.Markdown(
        "## Manual Audio Transcription Tool\nTranscribe each audio segment. `[INAUDIBLE]` if unclear. Audio stored in JSON."
    )

    audio_player = gr.Audio(first_audio, label="Audio Player", autoplay=True)
    transcription_box = gr.Textbox(
        value=first_transcript,
        label="Transcription",
        placeholder="Type transcription here...",
        lines=3,
        max_lines=5,
    )
    idx_state = gr.State(0)

    status_text = gr.Textbox(
        value=(
            f"Transcribing {os.path.basename(audio_files[start_idx])} ({start_idx+1}/{total_files})"
            if total_files > 0
            else "No audio files found."
        ),
        interactive=False,
    )

    with gr.Row():
        skip_btn = gr.Button("⏭ Skip / [INAUDIBLE]")
        next_btn = gr.Button("➡️ Next / Save")

    next_btn.click(
        fn=next_audio,
        inputs=[idx_state, transcription_box],
        outputs=[audio_player, status_text, idx_state, transcription_box],
    )
    skip_btn.click(
        fn=skip_audio,
        inputs=[idx_state],
        outputs=[audio_player, status_text, idx_state, transcription_box],
    )

demo.launch()
