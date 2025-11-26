import pandas as pd

df = pd.read_parquet("matched-00001.parquet")
print(df.columns)

original_form = df["candidate_text"]

# optional: convert to a DataFrame instead of Series
original_form_df = df[["candidate_text"]]

# Inspect
print(original_form.head())

original_form_df.to_csv("file4.csv", index=False, encoding="utf-8-sig")

# one_time_convert_csv_to_wav.py
# import pandas as pd
# import numpy as np
# import soundfile as sf
# import ast  # to parse string representations of lists

# # ----------------------------
# # CONFIG
# # ----------------------------
# CSV_FILE = "file2_audio.csv"  # your CSV file
# COLUMN_NAME = "audio_wav_bytes"  # column containing audio data
# OUTPUT_FOLDER = "./wav_output/"  # folder to save WAVs
# SAMPLE_RATE = 16000  # change if your audio has a different sample rate
# NUM_FILES = 5  # top N files to save

# # ----------------------------
# # CREATE OUTPUT FOLDER
# # ----------------------------
# import os


# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # ----------------------------
# # LOAD CSV
# # ----------------------------
# df = pd.read_csv(CSV_FILE)

# # ----------------------------
# # SAVE TOP N WAV FILES
# # ----------------------------
# for i, row in enumerate(df[COLUMN_NAME][:NUM_FILES]):
#     # If the CSV contains base64 encoded WAV bytes
#     try:
#         if isinstance(row, str) and row.startswith(
#             "Ukl"
#         ):  # common for base64 WAV starting with "RIFF" encoded
#             audio_bytes = base64.b64decode(row)
#         elif isinstance(row, bytes):
#             audio_bytes = row
#         elif isinstance(row, str):
#             # Maybe row is literal bytes string like "b'...'"
#             audio_bytes = eval(row)  # converts "b'RIFF...'" → bytes
#         else:
#             print(f"Skipping row {i}: unknown format")
#             continue

#         output_path = os.path.join(OUTPUT_FOLDER, f"audio_{i+1}.wav")
#         with open(output_path, "wb") as f:
#             f.write(audio_bytes)

#         print(f"Saved {output_path}")

#     except Exception as e:
#         print(f"Skipping row {i} due to error: {e}")

# print("✅ Done! Top 5 audio files saved.")
