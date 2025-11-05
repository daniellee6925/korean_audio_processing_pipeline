from splot_audio_v3 import SplitAudio


def main():
    processor = SplitAudio(config_path="config.yaml")
    processor.clear_sentence_folders()
    processor.process_all()
    processor.clear_temp_files()


if __name__ == "__main__":
    main()
