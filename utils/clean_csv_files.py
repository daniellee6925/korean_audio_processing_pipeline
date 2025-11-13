import os
from pathlib import Path


def clean_csv_without_wav(root_dir: str):
    root = Path(root_dir)
    csv_files = list(root.rglob("*.csv"))
    deleted_count = 0

    for csv_file in csv_files:
        parent_dir = csv_file.parent
        corresponding_wav = Path(os.path.join(parent_dir, f"{csv_file.stem}.wav"))
        # Check if any .wav file exists in the same folder

        if not corresponding_wav.exists():
            try:
                os.remove(csv_file)
                deleted_count += 1
            except Exception as e:
                print(f"couldn't delete {csv_file}: {e}")

    print(f"\nDone! Deleted {deleted_count} CSV files out of {len(csv_files)} without WAVs.")


def find_and_delete_segment_csvs(root_dir, filename="segment_all.csv", dry_run=True):
    """
    Recursively find and delete all files with the specified name.

    Args:
        root_dir: Root directory to search from
        filename: Name of the file to delete (default: "segment_all.csv")
        dry_run: If True, only show what would be deleted without actually deleting
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory not found at {root_dir}")
        return

    # Find all matching files recursively
    files_to_delete = list(root_path.rglob(filename))

    if not files_to_delete:
        print(f"No '{filename}' files found in {root_dir}")
        return

    print(f"Found {len(files_to_delete)} file(s) named '{filename}':\n")

    for file_path in files_to_delete:
        print(f"  {file_path}")

    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN MODE - No files were deleted")
        print("Set dry_run=False to actually delete these files")
        print(f"{'='*60}")
    else:
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"✓ Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {file_path}: {e}")

        print(f"\n{'='*60}")
        print(f"✓ Successfully deleted {deleted_count} file(s)")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Update this path to your root directory
    root_dir = "data/wavs_20250416_013301_segments"

    # First run in dry_run mode to see what would be deleted
    clean_csv_without_wav(root_dir)
