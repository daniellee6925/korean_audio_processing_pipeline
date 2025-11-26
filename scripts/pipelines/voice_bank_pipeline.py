from utils.move_text_files import TxtFileOrganizer
from utils.clean_empty_folders import delete_empty_deepest_folders
from split_text.split_text_period import TextSentenceSplitter
from split_audio.split_audio_thread import SplitAudio
from utils.delete_files import delete_files_by_extension
from utils.clean_file_name import CleanFileName
from filters.filter_files_by_length import FilterFilesByLength
from utils.delete_specific_files import delete_files
from utils.delete_files import delete_files_by_extension
from utils.merge_folders import DirectoryMerger
from audio_label.add_meta_data import process_root_dir_parallel
from audio_label.process import process_root_dir


def main(audio_dir: str, text_dir):
    processor = SplitAudio(
        root_dir=audio_dir,
        output_dir=audio_dir,
        min_silence_ms=300,
        min_segment_ms=1000,
        segment_subfolders=False,
        min_len=5.0,
    )
    processor.process_all()
    processor.clear_temp_files()

    delete_files_by_extension(root_dir=audio_dir, extension="csv")
    FilterFilesByLength(
        root_dir=audio_dir, file_format="wav", min_dur=5.0, max_dur=30.0
    ).process_all()

    delete_files(audio_dir, ".wav")


def post_process(audio_dir):
    merger = DirectoryMerger(
        dir1=audio_dir,
        dir2=f"{audio_dir}_trans",
        output_dir="",  # set if new directory needed
    )
    merger.merge(copy_mode=True)
    process_root_dir(audio_dir)
    delete_files(audio_dir, ".json")
    delete_files_by_extension(audio_dir, "txt")
    process_root_dir_parallel(audio_dir, 8)
    delete_empty_deepest_folders(audio_dir)


if __name__ == "__main__":
    # main(audio_dir="voice_bank/voice_casting", text_dir="TEXT_251114")
    post_process(audio_dir="voice_bank/voice_casting")
