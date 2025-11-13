import os
import shutil
import gradio as gr
from pathlib import Path

AUDIO_DIR = Path("keep_or_trash/audio_files")
KEEP_DIR = Path("keep_or_trash/keep")
DSCARD_DIR = Path("keep_or_trash/discard")

os.makedirs(KEEP_DIR, exist_ok=True)
os.makedirs(DSCARD_DIR, exist_ok=True)

# Get audio files
audio_files = []
for file_path in AUDIO_DIR.rglob("*"):
    if file_path.suffix.lower() in [".wav", ".mp3"]:
        rel_path = file_path.relative_to(AUDIO_DIR)
        audio_files.append(rel_path)
total_files = len(audio_files)
print(total_files)


def get_current_audio(idx):
    """Return the path to the current audio file, or None if finished."""
    if idx < total_files:
        return os.path.join(AUDIO_DIR, audio_files[idx])
    return None


def handle_keep_discard(action, idx):
    """
    action: 'keep' or 'discard'
    idx: current index into audio_files
    returns: (next_audio_path or None, status_text, new_idx)
    """
    if idx >= total_files:
        return None, f"✅ All {total_files} files processed.", idx

    current_rel_path = audio_files[idx]
    src = os.path.join(AUDIO_DIR, current_rel_path)
    if action == "keep":
        dst = os.path.join(KEEP_DIR, current_rel_path)
    else:
        dst = os.path.join(DSCARD_DIR, current_rel_path)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)

    new_idx = idx + 1
    processed_count = new_idx  # progress = index + 1

    if new_idx >= total_files:
        status = (
            f"✅ All {total_files} files processed ({processed_count}/{total_files})."
        )
        return None, status, new_idx
    else:
        next_file = audio_files[new_idx]
        status = f"Now playing {next_file} ({processed_count}/{total_files})."
        return get_current_audio(new_idx), status, new_idx


# Wrappers for buttons
def on_keep(idx):
    return handle_keep_discard("keep", idx)


def on_discard(idx):
    return handle_keep_discard("discard", idx)


# Gradio UI
with gr.Blocks(title="Audio Sorter", theme="soft") as demo:
    gr.Markdown("## Audio Sorting Tool\nKeep or discard audio files.")
    audio_player = gr.Audio(get_current_audio(0), label="Audio Player", autoplay=True)

    idx_state = gr.State(0)  # Only state needed

    status_text = gr.Textbox(
        value=(
            f"Total of {total_files} files to process"
            if audio_files
            else "No audio files found."
        ),
        interactive=False,
    )

    with gr.Row():
        keep_btn = gr.Button("✅ Keep")
        discard_btn = gr.Button("❌ Discard")

    keep_btn.click(
        fn=on_keep, inputs=[idx_state], outputs=[audio_player, status_text, idx_state]
    )

    discard_btn.click(
        fn=on_discard,
        inputs=[idx_state],
        outputs=[audio_player, status_text, idx_state],
    )

demo.launch()
