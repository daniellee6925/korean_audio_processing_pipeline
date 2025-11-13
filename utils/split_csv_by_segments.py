import csv
import os
from pathlib import Path


def split_csv_by_segments(csv_path, parent_dir):
    """
    Split a CSV file into individual segment CSVs in their respective folders.

    Args:
        csv_path: Path to the CSV file to split
        parent_dir: Parent directory containing segment folders
    """
    print(f"\nProcessing: {csv_path}")

    # Read the CSV file
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames

    # Group rows by segment
    segments = {}
    for row in rows:
        # Skip empty rows
        if not row.get("segment_folder"):
            continue

        # Extract segment number from folder path
        segment_folder = row["segment_folder"]
        segment_name = Path(segment_folder).name  # e.g., "segment_1"

        if segment_name not in segments:
            segments[segment_name] = []
        segments[segment_name].append(row)

    # Find existing segment folders
    existing_folders = set()
    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith("segment_"):
            existing_folders.add(item.name)

    print(f"  → Found {len(existing_folders)} existing segment folders")
    print(f"  → CSV contains {len(segments)} unique segments")

    # Write individual CSV files only for existing folders
    created_count = 0
    skipped_count = 0

    for segment_name, segment_rows in segments.items():
        # Create the segment folder path
        segment_folder = parent_dir / segment_name

        # Only create CSV if the folder exists
        if segment_name in existing_folders:
            # Create CSV file path
            csv_file = segment_folder / f"{segment_name}.csv"

            # Write the CSV file
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(segment_rows)

            created_count += 1
        else:
            skipped_count += 1

    print(f"  → Created {created_count} segment CSV files")
    if skipped_count > 0:
        print(f"  → Skipped {skipped_count} segments (folders don't exist)")

    return created_count


def process_root_directory(root_dir):
    """
    Process all CSV files in subdirectories of root_dir.

    Structure expected:
    root_dir/
        download_2_segment/
            segment_all.csv
            segment_1/
            segment_2/
            ...
        download_3_segment/
            segment_all.csv
            segment_1/
            ...

    Args:
        root_dir: Root directory containing subdirectories with CSV files
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory not found at {root_dir}")
        return

    # Find all CSV files in second level (subdirectories of root)
    csv_files = []
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            # Look for CSV files in this subdirectory
            for csv_file in subdir.glob("*.csv"):
                csv_files.append((csv_file, subdir))

    if not csv_files:
        print(f"No CSV files found in subdirectories of {root_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process\n")
    print("=" * 60)

    total_segments = 0
    for csv_file, parent_dir in csv_files:
        segments_created = split_csv_by_segments(csv_file, parent_dir)
        total_segments += segments_created

    print("=" * 60)
    print(f"\n✓ Done! Processed {len(csv_files)} CSV file(s)")
    print(f"✓ Created {total_segments} total segment CSV files")


if __name__ == "__main__":
    # Update this path to your root directory
    root_dir = "data/wavs_20250416_012741_splits_filtered"

    process_root_directory(root_dir)
