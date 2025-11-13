import os
import shutil
import gradio as gr
from pathlib import Path

AUDIO_DIR = Path("data/wavs_20250416_013301_segments")
TRASH_DIR = Path("gradio_ui/keep_or_trash/trash")

os.makedirs(TRASH_DIR, exist_ok=True)

# Get audio files
audio_files = []
for file_path in AUDIO_DIR.rglob("*"):
    if file_path.suffix.lower() in [".wav", ".mp3"]:
        rel_path = file_path.relative_to(AUDIO_DIR)
        audio_files.append(rel_path)
total_files = len(audio_files)
print(total_files)


def get_current_audio(idx, history):
    """Return the path to the current audio file, or None if finished."""
    if idx < total_files:
        file_path = audio_files[idx]
        # Check if file was previously discarded
        if file_path in history and history[file_path] == "discarded":
            # Skip to next
            return get_current_audio(idx + 1, history)
        return os.path.join(AUDIO_DIR, file_path)
    return None


def handle_keep(idx, history):
    """Move to next file without doing anything."""
    if idx >= total_files:
        return None, f"✅ All files processed.", idx, history

    new_idx = idx + 1

    # Skip any discarded files
    while (
        new_idx < total_files
        and audio_files[new_idx] in history
        and history[audio_files[new_idx]] == "discarded"
    ):
        new_idx += 1

    if new_idx >= total_files:
        status = f"✅ All files processed."
        return None, status, new_idx, history
    else:
        next_file = audio_files[new_idx]
        status = f"Now playing {next_file} ({new_idx + 1}/{total_files})."
        return get_current_audio(new_idx, history), status, new_idx, history


def handle_discard(idx, history):
    """Move file to trash and go to next."""
    if idx >= total_files:
        return None, f"✅ All files processed.", idx, history

    current_rel_path = audio_files[idx]
    src = os.path.join(AUDIO_DIR, current_rel_path)
    dst = os.path.join(TRASH_DIR, current_rel_path)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)

    # Update history
    history[current_rel_path] = "discarded"

    new_idx = idx + 1

    # Skip any discarded files
    while (
        new_idx < total_files
        and audio_files[new_idx] in history
        and history[audio_files[new_idx]] == "discarded"
    ):
        new_idx += 1

    if new_idx >= total_files:
        status = f"✅ All files processed."
        return None, status, new_idx, history
    else:
        next_file = audio_files[new_idx]
        status = f"Now playing {next_file} ({new_idx + 1}/{total_files})."
        return get_current_audio(new_idx, history), status, new_idx, history


def handle_undo(idx, history):
    """Go back to previous file and restore it if it was discarded."""
    if idx <= 0:
        # Already at first file
        status = "Already at the first file."
        return get_current_audio(0, history), status, idx, history

    new_idx = idx - 1
    prev_file = audio_files[new_idx]

    # If previous file was discarded, restore it
    if prev_file in history and history[prev_file] == "discarded":
        src = os.path.join(TRASH_DIR, prev_file)
        dst = os.path.join(AUDIO_DIR, prev_file)

        if os.path.exists(src):
            shutil.move(src, dst)
            del history[prev_file]

    status = f"Went back to {prev_file} ({new_idx + 1}/{total_files})."
    return get_current_audio(new_idx, history), status, new_idx, history


# Gradio UI
with gr.Blocks(title="Audio Discarder", theme="soft") as demo:
    gr.Markdown(
        "## Audio Discard Tool\nKeep or discard audio files. Discarded files can be undone."
    )
    audio_player = gr.Audio(get_current_audio(0, {}), label="Audio Player", autoplay=True)

    idx_state = gr.State(0)
    history_state = gr.State({})  # Track discarded files

    status_text = gr.Textbox(
        value=(
            f"Total of {total_files} files to process" if audio_files else "No audio files found."
        ),
        interactive=False,
    )

    with gr.Row():
        undo_btn = gr.Button("↩️ Undo")
        keep_btn = gr.Button("✅ Keep (Next)")
        discard_btn = gr.Button("❌ Discard")

    keep_btn.click(
        fn=handle_keep,
        inputs=[idx_state, history_state],
        outputs=[audio_player, status_text, idx_state, history_state],
    )

    discard_btn.click(
        fn=handle_discard,
        inputs=[idx_state, history_state],
        outputs=[audio_player, status_text, idx_state, history_state],
    )

    undo_btn.click(
        fn=handle_undo,
        inputs=[idx_state, history_state],
        outputs=[audio_player, status_text, idx_state, history_state],
    )

demo.launch()
