import os
import csv
import shutil
import re
from pathlib import Path
from loguru import logger
import yaml
from typing import Dict, Any, List
from tqdm import tqdm
from normalize_korean import KoreanTextNormalizer


class ValidateSplit:
    def __init__(self, config_path: str = "config.yaml"):
        # load config
        self.config = self.load_config(config_path)

        data_cfg = self.config["data"]
        val_cfg = self.config["val"]

        # file path configs
        self.root_dir = Path(data_cfg.get("root_dir", "/workspace/audio_files"))
        self.folder_prefix = data_cfg.get("folder_prefix", None)
        self.nested_folders = data_cfg.get("nested_folders", False)

        # validate configs
        self.ratio_threshold = val_cfg.get("ratio_threshold", 0.5)
        self.val_csv_output = val_cfg.get("val_csv_output", "validation_summary.csv")
        self.delete_files = val_cfg.get("delete_files", False)

        self.normalizer = KoreanTextNormalizer(overwrite_cache=False)

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def validate_folder(
        self,
        folder: Path,
        trans: List[Path],
        txts: List[Path],
        writer: csv.DictWriter,
    ):
        if len(trans) != len(txts):
            logger.debug(
                f"len of trans {len(trans)} and len of txts {(len(txts))} do not match: Automatic Fail"
            )
            writer.writerow(
                {
                    "folder": folder,
                    "num_sentences": "n/a",
                    "avg_ratio_diff": "sentence length mismatch - automatic fail",
                    "status": "FAIL",
                }
            )
            return
        txt_map = {t.stem: t for t in txts}
        matched = [
            (tr, txt_map[tr.stem.replace("trans_", "")])
            for tr in trans
            if tr.stem.replace("trans_", "") in txt_map
        ]

        if not matched:
            logger.warning(f"No matching audio/text pairs in {folder}")
            return

        ratio_diffs = []
        flagged_sentences = 0
        status = "PASS"
        for trans_file, text_file in matched:
            try:
                with trans_file.open("r") as f:
                    transcribed_text = f.read().strip()

                if re.findall(r"\d+", transcribed_text):
                    transcribed_text = self.normalizer.normalize_text(transcribed_text)

                with text_file.open("r") as f:
                    reference_text = f.read().strip()

                pred_count = len(transcribed_text.split())
                ref_count = len(reference_text.split())

                if ref_count == 0:
                    continue

                ratio_diff = abs(pred_count - ref_count) / ref_count
                ratio_diffs.append(ratio_diff)
                if ratio_diff > self.ratio_threshold:
                    flagged_sentences += 1
                    logger.debug(f"Ratio diff too high for {trans_file.name}: {ratio_diff:.3f}")
            except Exception as e:
                logger.error(f"Error processing {trans_file}: {e}")
                status = "FAIL"
        avg_ratio_diff = sum(ratio_diffs) / len(ratio_diffs) if ratio_diffs else 0.0
        status = "FAIL" if flagged_sentences > 2 else "PASS"
        writer.writerow(
            {
                "folder": folder,
                "num_sentences": len(matched),
                "avg_ratio_diff": f"{avg_ratio_diff:.3f}",
                "status": status,
            }
        )

    def iter_files(self):
        """Yield (dirpath, wav_files, txt_files) tuples where both exist."""
        for dirpath, _, files in os.walk(self.root_dir):
            dirpath = Path(dirpath)
            if self.folder_prefix is not None and not dirpath.name.startswith(self.folder_prefix):
                continue

            all_txts = sorted([dirpath / f for f in files if f.lower().endswith(".txt")])
            if not all_txts:
                continue

            trans = [p for p in all_txts if p.name.lower().startswith("trans_")]
            txts = [p for p in all_txts if not p.name.lower().startswith("trans_")]

            if trans and txts:
                yield dirpath, trans, txts

    def iter_nested_files(self):
        """Yield (dirpath, wav_files, txt_files) tuples where both exist."""
        root_path = Path(self.root_dir)
        results = {}

        for dir_path, dirnames, filenames in os.walk(root_path):
            dirpath = Path(dir_path)

            if not dirnames:
                parent_dir = dirpath.parent
                all_txts = [dirpath / f for f in filenames if f.lower().endswith(".txt")]
                if not all_txts:
                    continue

                trans_files = sorted([f for f in all_txts if f.name.lower().startswith("trans_")])
                txt_files = sorted([f for f in all_txts if not f.name.lower().startswith("trans_")])

                if trans_files or txt_files:
                    if parent_dir not in results:
                        results[parent_dir] = {"trans": [], "txt": []}
                    results[parent_dir]["trans"].extend(trans_files)
                    results[parent_dir]["txt"].extend(txt_files)
        return results

    def open_csv_writer(self, csv_name: str):
        """Open CSV and return a tuple (file_handle, writer)."""
        csv_path = self.root_dir / csv_name
        f = csv_path.open("w", encoding="utf-8", newline="")
        fieldnames = ["folder", "num_sentences", "avg_ratio_diff", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        return f, writer, csv_path

    @staticmethod
    def get_pairs(csv_path) -> list[tuple[str, str]]:
        try:
            pairs = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    folder = row["folder"]
                    status = row["status"]
                    pairs.append((folder, status))
            return pairs
        except FileNotFoundError:
            print(f"Error: File not found: {csv_path}")
            return []

    @staticmethod
    def summary(pairs) -> None:
        total = len(pairs)
        passed = sum(1 for _, status in pairs if status.upper() == "PASS")
        success_rate = passed / total * 100 if total > 0 else 0
        logger.success(f"Finished Validating {total} Folders.")
        logger.success(f"Total of {passed} Folders Passed with a pass-rate of {success_rate:.2f}%")

    @staticmethod
    def delete_fails(pairs) -> None:
        delete_paths = [Path(f) for f, status in pairs if status.upper() == "FAIL"]

        for folder_path in delete_paths:
            if folder_path.exists() and folder_path.is_dir():
                try:
                    shutil.rmtree(folder_path)  # delete parent in some cases
                except Exception as e:
                    logger.error(f"Failed to delete {folder_path}: {e}")
        logger.info(f"Total of {len(delete_paths)} failed Folders deleted.")

    def process_all(self) -> None:
        """
        Validate folders recursively. If folder_prefix is given, only process matching folders.
        """
        csv_name = (
            f"{self.folder_prefix}_{self.val_csv_output}"
            if self.folder_prefix
            else self.val_csv_output
        )
        csv_file, writer, csv_path = self.open_csv_writer(csv_name)
        logger.info(f"Starting validation under {self.root_dir} (prefix={self.folder_prefix})")

        folders_to_validate = self.iter_nested_files()

        if not folders_to_validate:
            logger.warning(f"No folders found matching prefix '{self.folder_prefix}'")
            return

        logger.info(f"Found {len(folders_to_validate)} folders to validate.")

        try:
            for dirpath, files in tqdm(
                folders_to_validate.items(), total=len(folders_to_validate), desc="Validating"
            ):
                trans_files = files["trans"]
                txt_files = files["txt"]
                logger.info(f"Validating folder: {dirpath}")
                try:
                    self.validate_folder(dirpath, trans_files, txt_files, writer)
                except Exception as e:
                    logger.error(f"Error validating folder {dirpath}: {e}")

            logger.success(f"Validation finished. Results saved to: {csv_path}")
        finally:
            csv_file.close()

        pairs = self.get_pairs(csv_path)
        self.summary(pairs)

        if self.delete_files:
            self.delete_fails(pairs)


if __name__ == "__main__":
    validator = ValidateSplit(config_path="configs/val_config.yaml")
    validator.process_all()
