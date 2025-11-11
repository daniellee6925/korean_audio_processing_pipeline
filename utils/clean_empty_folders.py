from pathlib import Path
import shutil


def remove_empty_non_wav_folders(root_dir: str):
    """
    Remove all subfolders under root_dir that do not contain any .wav files.
    Returns a summary dictionary with counts and deleted folder paths.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")

    deleted_folders = []
    kept_folders = []

    # rglob for directories only (bottom-up traversal is safer for deletion)
    for folder in sorted(root_path.rglob("*"), reverse=True):
        if folder.is_dir():
            # Check if folder contains any .wav files (recursively only in this folder, not subfolders)
            has_wav = any(f.suffix.lower() == ".wav" for f in folder.iterdir() if f.is_file())

            if not has_wav:
                shutil.rmtree(folder)
                deleted_folders.append(str(folder))
            else:
                kept_folders.append(str(folder))

    # Build summary
    summary = {
        "root_dir": str(root_path),
        "total_folders_scanned": len(deleted_folders) + len(kept_folders),
        "folders_deleted": len(deleted_folders),
        "folders_kept": len(kept_folders),
        "deleted_folder_paths": deleted_folders,
        "kept_folder_paths": kept_folders,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("REMOVE NON-WAV FOLDER SUMMARY")
    print("=" * 50)
    print(f"Root Directory: {summary['root_dir']}")
    print(f"Total Folders Scanned: {summary['total_folders_scanned']}")
    print(f"Folders Deleted: {summary['folders_deleted']}")
    print(f"Folders Kept: {summary['folders_kept']}")
    print("=" * 50 + "\n")

    return summary


if __name__ == "__main__":
    summary = remove_empty_non_wav_folders("voice_casting")
