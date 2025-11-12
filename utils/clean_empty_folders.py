from pathlib import Path
import shutil


def remove_all_empty_folders(root_dir: str, extension: str):
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
            has_wav = any(
                f.suffix.lower() == f".{extension}" for f in folder.iterdir() if f.is_file()
            )

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


def remove_folders_without_segments(root_dir: str):
    """
    Deletes all folders under `root_dir` that do NOT contain any subfolder
    starting with 'segment_'.

    Args:
        root_dir (str): Path to the root directory.

    Returns:
        dict: Summary containing counts and deleted folder paths.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")

    deleted_folders = []
    kept_folders = []

    # Scan all folders (not including root)
    all_folders = [f for f in root_path.rglob("*") if f.is_dir()]

    for folder in all_folders:
        # Skip folders that are themselves segment folders
        if folder.name.startswith("segment_"):
            kept_folders.append(str(folder))
            continue

        # Check if folder contains any subfolder starting with segment_
        has_segment_subdir = any(
            sub.is_dir() and sub.name.startswith("segment_") for sub in folder.iterdir()
        )

        if not has_segment_subdir:
            try:
                shutil.rmtree(folder)
                deleted_folders.append(str(folder))
            except Exception as e:
                print(f"⚠️ Failed to delete {folder}: {e}")
        else:
            kept_folders.append(str(folder))

    summary = {
        "root_dir": str(root_path),
        "folders_scanned": len(all_folders),
        "folders_deleted": len(deleted_folders),
        "folders_kept": len(kept_folders),
        "deleted_folder_paths": deleted_folders,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("REMOVE FOLDERS WITHOUT SEGMENT_ SUBDIR SUMMARY")
    print("=" * 50)
    print(f"Root Directory: {summary['root_dir']}")
    print(f"Folders Scanned: {summary['folders_scanned']}")
    print(f"Folders Deleted: {summary['folders_deleted']}")
    print(f"Folders Kept: {summary['folders_kept']}")
    print("=" * 50 + "\n")

    return summary


def remove_empty_folders_max_depth(root_dir: str, extension: str):
    """
    Remove only the deepest (max-depth) subfolders under root_dir that do not contain any .wav files.
    Returns a summary dictionary with counts and deleted folder paths.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")

    deleted_folders = []
    kept_folders = []

    # Get all folders
    all_folders = [f for f in root_path.rglob("*") if f.is_dir()]
    if not all_folders:
        print("No subfolders found.")
        return

    # Find maximum depth
    max_depth = max(len(folder.relative_to(root_path).parts) for folder in all_folders)

    # Filter folders at max depth
    deepest_folders = [f for f in all_folders if len(f.relative_to(root_path).parts) == max_depth]

    # Process only those deepest folders
    for folder in deepest_folders:
        has_wav = any(f.suffix.lower() == f".{extension}" for f in folder.iterdir() if f.is_file())
        if not has_wav:
            shutil.rmtree(folder)
            deleted_folders.append(str(folder))
        else:
            kept_folders.append(str(folder))

    # Build summary
    summary = {
        "root_dir": str(root_path),
        "max_depth": max_depth,
        "total_folders_scanned": len(deepest_folders),
        "folders_deleted": len(deleted_folders),
        "folders_kept": len(kept_folders),
        "deleted_folder_paths": deleted_folders,
        "kept_folder_paths": kept_folders,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("REMOVE MAX-DEPTH NON-WAV FOLDER SUMMARY")
    print("=" * 50)
    print(f"Root Directory: {summary['root_dir']}")
    print(f"Max Depth: {summary['max_depth']}")
    print(f"Folders Scanned (Max Depth Only): {summary['total_folders_scanned']}")
    print(f"Folders Deleted: {summary['folders_deleted']}")
    print(f"Folders Kept: {summary['folders_kept']}")
    print("=" * 50 + "\n")

    return summary


if __name__ == "__main__":
    # summary = remove_empty_folders_max_depth("data/wavs_20250416_012741_splits", extension="wav")
    summary = remove_folders_without_segments("data/wavs_20250416_012741_splits copy")
