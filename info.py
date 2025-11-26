from summary.summarize_by_base import AudioDirectorySummary


def main(root_dir: str):
    summarizer = AudioDirectorySummary(root_dir=root_dir, deepest=True)

    # Generate summary
    summary = summarizer.generate_summary()


if __name__ == "__main__":
    main(root_dir="podbbang")
