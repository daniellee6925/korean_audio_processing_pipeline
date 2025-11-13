import os
import shutil


AUDIO_DIR = "audio_files"
KEEP_DIR = "keep"
DSCARD_DIR = "discard"


def revert():
    for folder in [KEEP_DIR, DSCARD_DIR]:
        for f in os.listdir(folder):
            src = os.path.join(folder, f)
            dst = os.path.join(AUDIO_DIR, f)
            shutil.move(src, dst)


if __name__ == "__main__":
    revert()
