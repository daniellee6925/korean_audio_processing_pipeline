from utils.move_text_files import TxtFileOrganizer
from utils.clean_empty_folders import remove_empty_folders_max_depth
from split_text.split_text_period import TextSentenceSplitter
from split_audio.split_audio_long_base import SplitAudio
from utils.delete_files import delete_files_by_extension
from utils.clean_file_name import CleanFileName


def main(audio_dir: str, text_dir):
    organizer = TxtFileOrganizer(
        txt_root=text_dir,
        folder_root=audio_dir,
    )
    organizer.organize()

    CleanFileName(
        root_dir=audio_dir,
        extensions=["WAV"],
        original="_Tr1",
        change_to="",
        portion="any",
    ).process_all()

    CleanFileName(
        root_dir=text_dir,
        extensions=["txt"],
        original="",
        change_to="",
        portion="suffix3",
    ).process_all()

    summary = remove_empty_folders_max_depth(audio_dir, extension="txt")
    splitter = TextSentenceSplitter(
        root_dir=audio_dir, sentence_folders=True  # toggle this to True/False
    )
    splitter.process_directory()

    processor = SplitAudio(
        root_dir=audio_dir,
        output_dir=audio_dir,
        min_silence_ms=1500,
        min_segment_ms=1000,
        segment_subfolders=True,
    )
    processor.process_all()
    processor.clear_temp_files()

    delete_files_by_extension(root_dir=audio_dir, extension="csv")


if __name__ == "__main__":
    main(audio_dir="Recording_251114", text_dir="TEXT_251114")
