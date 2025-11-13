import json
import base64
import os

# Path to your JSON file
JSON_FILE = "metadata.json"
OUTPUT_DIR = "decoded_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Decode each audio segment
for key, entry in data.items():
    audio_b64 = entry.get("audio_segment")
    if not audio_b64:
        print(f"No audio found for {key}")
        continue

    # Decode base64
    audio_bytes = base64.b64decode(audio_b64)

    # Determine output file path
    folder_name = entry.get("tar_file", "unknown_folder")
    file_name = entry.get("wav_file", "audio.wav")
    out_folder = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, file_name)

    # Write decoded audio
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    print(f"Decoded audio saved to {out_path}")

print("All audio segments decoded successfully.")
