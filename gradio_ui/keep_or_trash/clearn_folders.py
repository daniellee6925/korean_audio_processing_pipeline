import os
import shutil


def clean_folders():
    """Remove all content from specified directories."""
    directories = ["audio_files", "discard", "keep", ".trash"]

    for directory in directories:
        if os.path.exists(directory):
            # Remove all contents
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    print(f"✓ Removed: {item_path}")
                except Exception as e:
                    print(f"✗ Failed to delete {item_path}: {e}")
            print(f"✓ Cleaned directory: {directory}")
        else:
            print(f"⊘ Directory not found: {directory}")

    print("\n✅ Cleanup complete!")


if __name__ == "__main__":
    # Confirmation prompt for safety
    response = input(
        "⚠️  This will DELETE ALL FILES in audio_files, discard, keep, and .trash directories.\nAre you sure? (yes/no): "
    )

    if response.lower() in ["yes", "y"]:
        clean_folders()
    else:
        print("Cleanup cancelled.")
