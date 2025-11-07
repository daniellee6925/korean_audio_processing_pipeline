from splot_audio_v3 import SplitAudio


def main():
    processor = SplitAudio(config_path="config.yaml")
    processor.clear_sentence_folders()
    processor.process_all()
    # processor.process_file(
    #     wav_path="sentence_14/160101_014_Tr1.wav", save_path="sentence_14"
    # )
    processor.clear_temp_files()


if __name__ == "__main__":
    main()
