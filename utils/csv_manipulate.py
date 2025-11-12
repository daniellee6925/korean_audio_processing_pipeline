import csv
from pathlib import Path


def process_csv_files(folder_path: str):
    folder_path = Path(folder_path)
    csv_files = list(folder_path.rglob("*.csv"))

    for csv_file in csv_files:
        updated_rows = []

        with open(csv_file, "r", encoding="utf-8") as f:
            headers = ["segment_folder", "segment_file", "start_sec", "end_sec", "duration_sec"]
            reader = csv.DictReader(f, fieldnames=headers)
            next(reader)  # skip header row

            for row in reader:
                if not any(row.values()):
                    continue

                seg_file_path = Path(row["segment_file"])
                print(seg_file_path)

                file_name = seg_file_path.parts[1].replace("_segments", ".wav")
                folder_name = seg_file_path.parts[0]

                row["segment_folder"] = folder_name
                row["segment_file"] = file_name

                updated_rows.append(row)
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(updated_rows)

        print(f"Processed {csv_file} → {len(updated_rows)} rows")


if __name__ == "__main__":
    process_csv_files("Korean_Conversational_Speech_Corpus")
