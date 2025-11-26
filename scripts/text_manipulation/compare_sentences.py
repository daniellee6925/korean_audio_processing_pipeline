from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import csv
import pandas as pd


model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")  # Korean SBERT


def cosine_similarity(sent1: str, sent2: str, threshold: float = 0.75) -> bool:
    """
    Compare two Korean sentences and flag if they are semantically different.

    Args:
        sent1, sent2: Sentences to compare
        threshold: Cosine similarity threshold below which sentences are considered different

    Returns:
        True if sentences are semantically different, False otherwise
    """
    # Encode sentences
    embeddings = model.encode([sent1, sent2], convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Flag as different if similarity below threshold
    return similarity


def compare_csv_batch(
    input_path: Path,
    output_path: Path,
    col1: str,
    col2: str,
    threshold: float = 0.9,
    batch_size: int = 32,
):
    """
    Compare two columns of Korean sentences in a CSV and output a new CSV with a 'different' flag.
    """

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    sentences1 = df[col1].tolist()
    sentences2 = df[col2].tolist()

    similarities = []

    for i in range(0, len(sentences1), batch_size):
        batch1 = sentences1[i : i + batch_size]
        batch2 = sentences2[i : i + batch_size]
        embeddings1 = model.encode(batch1, convert_to_tensor=True)
        embeddings2 = model.encode(batch2, convert_to_tensor=True)

        batch_sim = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().tolist()

        similarities.extend(batch_sim)

    df["similarity"] = similarities
    df["flagged"] = ["FLAG" if s < threshold else "" for s in similarities]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Processed {input_path.name} → {output_path.name}")


def compare_csv(input_path: str, output_path: str, col1: str, col2: str, threshold: float = 0.9):
    """
    Compare two columns of Korean sentences in a CSV and output a new CSV with a 'different' flag.
    """
    try:
        with open(input_path, "r", encoding="utf-8-sig") as f_in, open(
            output_path, "w", encoding="utf-8-sig", newline=""
        ) as f_out:

            reader = csv.DictReader(f_in)
            fieldnames = list(reader.fieldnames) + ["similarity", "flagged"]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                similarity_score = cosine_similarity(row[col1], row[col2])
                row["similarity"] = similarity_score
                row["flagged"] = "FLAG" if similarity_score < threshold else ""
                writer.writerow(row)

        print(f"Processed {input_path.name} → {output_path.name}")
    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {e}")


def process_all(root_dir: str, save_dir: str):
    root_path = Path(root_dir)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    csv_files = list(root_path.rglob("*.csv"))
    if not csv_files:
        print(f"No csv files found in {root_path}")
        return
    for csv_file in csv_files:
        output_path = save_path / csv_file.name
        compare_csv_batch(
            input_path=csv_file,
            output_path=output_path,
            col1="text",
            col2="transcribed",
            threshold=0.5,
        )

    print(f"Processed {len(csv_files)}")


if __name__ == "__main__":
    process_all("Merged", "Comparison")
