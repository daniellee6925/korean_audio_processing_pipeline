from summary.summarize_by_base import AudioDirectorySummary


def main(root_dir: str):
    summarizer = AudioDirectorySummary(root_dir=root_dir)

    # Generate summary
    summary = summarizer.generate_summary()

    # Access specific statistics
    print(f"\nQuick Stats:")
    print(f"Total audio time: {summary['total_duration_formatted']}")
    print(f"Total files: {summary['total_files']}")
    print("\n")


if __name__ == "__main__":
    main(root_dir="data/wavs_20250416_012741_splits")
