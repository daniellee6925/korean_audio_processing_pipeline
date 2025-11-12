import os
from pathlib import Path
from typing import List


class FolderJsonDeleter:
    """
    Deletes JSON files that are named after their parent folder.
    For example: voice_001/voice_001.json will be deleted.
    """

    def __init__(self):
        """Initialize the deleter."""
        pass

    def find_folder_named_jsons(self, root_dir: str) -> List[Path]:
        """
        Find all JSON files that are named after their parent folder.

        Args:
            root_dir: Root directory to search

        Returns:
            List of JSON file paths that match the folder name
        """
        root_path = Path(root_dir)
        matching_jsons = []

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        # Iterate through subdirectories
        for subdir in root_path.iterdir():
            if not subdir.is_dir():
                continue

            folder_name = subdir.name
            json_path = subdir / f"{folder_name}.json"

            if json_path.exists() and json_path.is_file():
                matching_jsons.append(json_path)

        return matching_jsons

    def delete_folder_jsons(self, root_dir: str, dry_run: bool = True) -> dict:
        """
        Delete all JSON files named after their parent folder.

        Args:
            root_dir: Root directory containing subdirectories
            dry_run: If True, only show what would be deleted without actually deleting

        Returns:
            Dictionary with statistics (found, deleted, errors)
        """
        print(f"Scanning directory: {root_dir}")
        print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'DELETE MODE'}")
        print("=" * 60)

        # Find all matching JSON files
        matching_jsons = self.find_folder_named_jsons(root_dir)

        deleted_count = 0
        error_count = 0
        errors = []

        if not matching_jsons:
            print("\nNo folder-named JSON files found.")
            return {"found": 0, "deleted": 0, "errors": 0, "error_details": []}

        print(f"\nFound {len(matching_jsons)} folder-named JSON files:")
        print("-" * 60)

        for json_path in matching_jsons:
            folder_name = json_path.parent.name
            relative_path = json_path.relative_to(root_dir)

            if dry_run:
                print(f"  [DRY RUN] Would delete: {relative_path}")
                deleted_count += 1
            else:
                try:
                    json_path.unlink()
                    print(f"  ✓ Deleted: {relative_path}")
                    deleted_count += 1
                except Exception as e:
                    error_count += 1
                    error_msg = f"Failed to delete {relative_path}: {e}"
                    errors.append(error_msg)
                    print(f"  ✗ {error_msg}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Files found: {len(matching_jsons)}")

        if dry_run:
            print(f"Files that would be deleted: {deleted_count}")
        else:
            print(f"Files deleted: {deleted_count}")
            print(f"Errors: {error_count}")

        return {
            "found": len(matching_jsons),
            "deleted": deleted_count,
            "errors": error_count,
            "error_details": errors,
        }

    def interactive_delete(self, root_dir: str):
        """
        Interactive mode that shows files and asks for confirmation before deleting.

        Args:
            root_dir: Root directory containing subdirectories
        """
        print("INTERACTIVE DELETE MODE")
        print("=" * 60)

        # First do a dry run to show what would be deleted
        stats = self.delete_folder_jsons(root_dir, dry_run=True)

        if stats["found"] == 0:
            return

        print("\n" + "=" * 60)
        response = (
            input(f"\nDo you want to delete these {stats['found']} files? (yes/no): ")
            .strip()
            .lower()
        )

        if response in ["yes", "y"]:
            print("\nProceeding with deletion...")
            self.delete_folder_jsons(root_dir, dry_run=False)
        else:
            print("\nDeletion cancelled.")


if __name__ == "__main__":
    deleter = FolderJsonDeleter()

    # Option 1: Dry run (safe, just shows what would be deleted)
    deleter.delete_folder_jsons(root_dir="data/kmong", dry_run=True)
