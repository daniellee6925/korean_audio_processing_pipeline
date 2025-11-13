from filters.filter_corrupt_files import FilterCorrupt
from filters.filter_files_by_length import FilterFilesByLength
from filters.filter_files_by_bgm import FilterByBGM
from filters.random_filter import keep_random_segments
from utils.clean_empty_folders import remove_empty_folders_max_depth
from split_audio.split_audio_long_base import SplitAudio
from utils.delete_files import delete_files_by_extension
from utils.clean_csv_files import clean_csv_without_wav
from summary.summarize_by_base import AudioDirectorySummary


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
    # # # remove corrupt files
    # # FilterCorrupt(root_dir=audio_dir, delete_bad=True).process_all()

    # # # remove long files
    # # FilterFilesByLength(
    # #     root_dir=audio_dir, file_format="wav", min_dur=5.0, max_dur=4000.0
    # # ).process_all()

    # filter files with bgm
    # FilterByBGM(root_dir=audio_dir).process_all()

    # split by segments
    processor = SplitAudio(
        root_dir=audio_dir, output_dir=f"{audio_dir}_segments", min_segment_ms=200
    )
    processor.process_all()
    processor.clear_temp_files()

    # keep short files only
    FilterFilesByLength(
        root_dir=f"{audio_dir}_segments", file_format="wav", min_dur=3.0, max_dur=12.0
    ).process_all()

    # filter files with bgm
    FilterByBGM(root_dir=f"{audio_dir}_segments")

    # filter to keep only a few
    keep_random_segments(root_dir=f"{audio_dir}_segments", keep_count=5, keep_type="wav")

    # remove empty folders
    summary = remove_empty_folders_max_depth(f"{audio_dir}_segments", extension="wav")

    clean_csv_without_wav(root_dir=f"{audio_dir}_segments")


def post_filter(audio_dir: str):
    clean_csv_without_wav(root_dir=f"{audio_dir}_segments")
    summary = remove_empty_folders_max_depth(f"{audio_dir}_segments", extension="wav")
    AudioDirectorySummary(root_dir=f"{audio_dir}_segments").generate_summary()


if __name__ == "__main__":
    # main(audio_dir="data/wavs_20250416_013301")
    post_filter(audio_dir="data/wavs_20250416_013301")
