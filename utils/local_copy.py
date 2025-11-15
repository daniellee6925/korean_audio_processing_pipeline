from pathlib import Path
from typing import List
from datetime import datetime


class RsyncCopier:
    def __init__(self, remote_user_host: str, remote_path: str, local_path: str):
        """
        Args:
            remote_user_host (str): Remote user and host, e.g., 'daniel@a100-4'
            remote_path (str): Remote directory path
            local_path (str): Local directory path to copy into
        """
        self.remote_user_host = remote_user_host
        self.remote_path = remote_path.rstrip("/")
        self.local_path = Path(local_path)
        self.local_path.mkdir(parents=True, exist_ok=True)

    def process_all(self, include_types: List[str], modified_after: datetime = None):
        """
        Copies files from remote_path to local_path, preserving the top-level folder.

        Args:
            include_types (List[str]): File extensions or patterns to include, e.g., ['*.wav', '*.csv']
            modified_after (datetime, optional): Only copy files modified after this datetime
        """
        if not include_types:
            raise ValueError("You must provide at least one file type to include.")

        include_patterns = " ".join(f"--include='{pattern}'" for pattern in include_types)
        include_dirs = "--include='*/'"  # preserve directory structure
        exclude_all = "--exclude='*'"

        update_flag = "--update" if modified_after else ""

        cmd = (
            f"rsync -avz {update_flag} {include_dirs} {include_patterns} {exclude_all} "
            f"{self.remote_user_host}:{self.remote_path} {self.local_path}/"
        )

        print(cmd)


# Example usage
if __name__ == "__main__":
    copier = RsyncCopier(
        remote_user_host="daniel@a100-4",
        remote_path="/fsx/workspace/daniel/korean_audio_processing_pipeline/Recording_251114",
        local_path="/Users/daniel/desktop/projects/local_copy",
    )

    # Copy only .txt files modified after Nov 1, 2025
    # ["*.txt"]
    copier.process_all(include_types=["*.txt"], modified_after=datetime(2025, 11, 1))
