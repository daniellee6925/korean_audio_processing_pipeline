import os
import json
import soundfile as sf  # pip install soundfile
from tqdm import tqdm
from loguru import logger

ROOT_DIR = "Voice_Bank/voice_pick"


def update_json_for_audio(audio_path, base_json_path, transcription_folder):
    # Load base JSON
    with open(base_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Update filepath to the segment file path ---
    data["filepath"] = audio_path  # full path
    # OR if you want relative:
    # data["filepath"] = os.path.relpath(audio_path, ROOT_DIR)

    # Recalculate audio length
    audio_data, samplerate = sf.read(audio_path)
    audio_length = len(audio_data) / samplerate
    data["audio_length"] = round(audio_length, 2)

    # Add transcription
    txt_file = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
    txt_path = os.path.join(transcription_folder, txt_file)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            transcription = f.read().strip()
        data["transcription"] = transcription
    else:
        data["transcription"] = ""

    # Save new JSON alongside audio segment
    output_json_path = os.path.splitext(audio_path)[0] + ".json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_root_dir(root_dir):
    for voice_folder in tqdm(os.listdir(root_dir), desc="Processing Json File"):
        voice_path = os.path.join(root_dir, voice_folder)
        if not os.path.isdir(voice_path):
            continue

        # Look for segment folders
        for item in os.listdir(voice_path):
            if item.endswith("_segment"):
                segment_folder = os.path.join(voice_path, item)
                base_json_name = item.replace("_segment", ".json")
                base_json_path = os.path.join(voice_path, base_json_name)
                if not os.path.exists(base_json_path):
                    logger.debug(f"Base JSON not found: {base_json_path}, skipping...")
                    continue

                # Process each audio file in the segment folder
                for file in os.listdir(segment_folder):
                    if file.endswith(".wav"):
                        audio_path = os.path.join(segment_folder, file)
                        update_json_for_audio(audio_path, base_json_path, segment_folder)
            logger.info(f"Finished Processing {item}")
    logger.success("Finished Processing Json Files")


if __name__ == "__main__":
    process_root_dir(ROOT_DIR)
