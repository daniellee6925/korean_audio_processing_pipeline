from pathlib import Path
import shutil

TRASH_DIR = Path("trash")
ORIGINAL_DIR = Path("keep_or_trash/audio_files")


def revert(trash_dir: Path, original_dir: Path):
    """
    Move all files from trash_dir back to their original locations under original_dir.
    """
    if not trash_dir.exists():
        print(f"No trash folder found at {trash_dir}")
        return

    for file_path in trash_dir.rglob("*"):
        if file_path.is_file():
            # Compute relative path inside trash
            rel_path = file_path.relative_to(trash_dir)
            # Destination path
            dest_path = original_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Move file back
            shutil.move(str(file_path), str(dest_path))
            print(f"Restored: {file_path} → {dest_path}")

    print("All files restored from trash!")


if __name__ == "__main__":
    revert(trash_dir=TRASH_DIR, original_dir=ORIGINAL_DIR)
