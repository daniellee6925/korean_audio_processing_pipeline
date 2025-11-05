import os


def find_folders(root_dir: str) -> list[str]:
    """Return a list of all subfolders in a root directory."""
    try:
        return [
            f
            for f in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, f))
        ]
    except FileNotFoundError:
        print(f"Warning: Root directory '{root_dir}' not found.")
        return []
    except PermissionError:
        print(f"Warning: Permission denied for '{root_dir}'.")
        return []


def find_files(folder_path: str, extension: str) -> list[str]:
    """
    Return a list of files in a folder matching the given extension.

    Args:
        folder_path: Path to the folder to search in.
        extension: File extension to look for (e.g., '.txt', '.wav').

    Returns:
        List of file paths matching the extension.
    """
    try:
        return [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(extension.lower())
        ]
    except FileNotFoundError:
        print(f"Warning: Folder '{folder_path}' not found.")
        return []
    except PermissionError:
        print(f"Warning: Permission denied for folder '{folder_path}'.")
        return []
    except Exception as e:
        print(f"Warning: Could not list files in '{folder_path}': {e}")
        return []


def make_dir(folder_path: str) -> None:
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(folder_path, exist_ok=True)
    except PermissionError:
        print(f"Warning: Permission denied while creating directory '{folder_path}'")
    except Exception as e:
        print(f"Warning: Could not create directory '{folder_path}': {e}")
