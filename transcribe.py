import whisper
import torch
import os

root_dir = "/workspace/archive"

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["WHISPER_CACHE_DIR"] = "/fsx/models"

model = whisper.load_model("large-v3", device=device)


for i, folder in enumerate(sorted(os.listdir(root_dir))):
    folder_path = os.path.join(root_dir, folder)
    wav_path = os.path.join(folder_path, "Tr1.WAV")

    if os.path.isfile(wav_path):
        print(f"Transcribing {wav_path}...")
        result = model.transcribe(wav_path)
        out_path = os.path.join(folder_path, f"1028_{i+2:03}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Saved: {out_path}")
