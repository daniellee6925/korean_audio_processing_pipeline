import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from filters.filter_corrupt_files import FilterCorrupt, FilterCorruptSegments
from filters.filter_files_by_length import FilterFilesByLength
from filters.filter_files_by_bgm import FilterByBGM
from utils.merge_dir import DirectoryMerger
from utils.summarize import AudioDirectorySummary
from audio_manipulation.split_audio_process import SplitAudio
from pipelines.difficult_sentences import AcousticDifficultyFilter
from utils.clean import Cleaner, CleanTask


def initial_filter(audio_dir: str):
    # remove corrupt files
    FilterCorrupt(root_dir=audio_dir, delete_bad=True).process_all()

    # remove long files
    FilterFilesByLength(
        root_dir=audio_dir, file_format="wav", min_dur=5.0, max_dur=4000.0
    ).process_all()

    # filter files with bgm
    FilterByBGM(root_dir=audio_dir).process_all()


def main(audio_dir: str):
    # remove corrupt files
    FilterCorrupt(root_dir=audio_dir, delete_bad=True).process_all()

    # split by segments
    processor = SplitAudio(
        root_dir=audio_dir,
        output_dir=f"{audio_dir}_segments",
        min_segment_ms=150,
        min_len=6.0,
        max_workers=8,
        batch_size=20,
    )
    processor.process_all()
    processor.clear_temp_files()

    FilterCorruptSegments(f"{audio_dir}_segments").process_all()
    # keep short files only
    FilterFilesByLength(
        root_dir=f"{audio_dir}_segments", file_format="wav", min_dur=5.0, max_dur=12.0
    ).process_all()

    # filter files with bgm
    FilterByBGM(root_dir=f"{audio_dir}_segments")

    # filter to keep only a few
    # skeep_random_segments(root_dir=f"{audio_dir}_segments", keep_count=10, keep_type="wav")

    # remove empty folders
    clean_csv_without_wav_folder(root_dir=f"{audio_dir}_segments")
    summary = delete_empty_deepest_folders(f"{audio_dir}_segments")
    AudioDirectorySummary(root_dir=f"{audio_dir}_segments").generate_summary()


def post_filter(audio_dir: str):
    FilterCorruptSegments(f"{audio_dir}_segments").process_all()
    clean_csv_without_wav_folder(root_dir=f"{audio_dir}_segments")
    summary = delete_empty_deepest_folders(f"{audio_dir}_segments")
    AudioDirectorySummary(root_dir=f"{audio_dir}_segments").generate_summary()


def post_transcribe(audio_dir: str):
    DirectoryMerger(
        dir1=audio_dir,
        dir2=f"{audio_dir}_trans",
        output_dir="",  # set if new directory needed
    ).merge(copy_mode=True)

    AcousticDifficultyFilter(root_dir=audio_dir).process_all()
    clean_csv_without_wav_folder(root_dir=f"{audio_dir}")
    summary = delete_empty_deepest_folders(f"{audio_dir}")
    AudioDirectorySummary(root_dir=f"{audio_dir}").generate_summary()


def post_filter(audio_dir: str):
    clean_files_without_wav(root_dir=f"{audio_dir}", extension="txt")
    clean_csv_without_wav_folder(root_dir=f"{audio_dir}")
    summary = delete_empty_deepest_folders(f"{audio_dir}")
    AudioDirectorySummary(root_dir=f"{audio_dir}").generate_summary()


def post_label(audio_dir: str):
    tasks = [
        CleanTask(action="clean_files", key_extension="json", query_extension="wav"),
        CleanTask(action="clean_files", key_extension="json", query_extension="txt"),
        CleanTask(action="delete_by_extension", query_extension="csv"),
        CleanTask(action="delete_empty_dirs"),
    ]
    Cleaner.process_all(root_dir=audio_dir, tasks=tasks)


if __name__ == "__main__":
    audio_dir = "poddbang_wavs/wavs_20250416_133945_segments"
    final = "Podbbang"
    # main(audio_dir)
    # post_transcribe(audio_dir)
    post_label(final)
