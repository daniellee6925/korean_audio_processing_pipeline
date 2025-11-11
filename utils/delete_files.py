import os


def delete_files_by_extension(root_dir: str, extension: str):
    """Recursively delete all files with the given extension under root_dir."""
    extension = extension.lower().lstrip(".")
    deleted_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(f".{extension}"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\n✅ Finished — deleted {deleted_count} '.{extension}' files.")


if __name__ == "__main__":
    delete_files_by_extension("voice_pick", "json")
