import os
import csv
import torch
import whisper
from pathlib import Path
from loguru import logger
import json
import yaml
from typing import Dict, Any, List


class ValidateSplit:
    def __init__(self, config_path: str = "config.yaml"):
        # load config
        self.config = self.load_config(config_path)

        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        val_cfg = self.config["val"]

        self.device = model_cfg.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # model configs
        self.model = whisper.load_model(model_cfg["name"], device=self.device)

        # file path configs
        self.root_dir = Path(data_cfg.get("root_dir", "/workspace/audio_files"))
        self.folder_prefix = data_cfg.get("folder_prefix", None)
        self.extension = data_cfg.get("extension", ".wav")
        self.output_prefix = data_cfg.get("output_prefix", "out_")

        self.transcribe_args = self.config.get("transcription", {})

        # validate configs
        self.ratio_threshold = val_cfg.get("ratio_threshold", 0.2)
        self.val_csv_output = val_cfg.get(
            "val_csv_output", "validation_summary.csv"
        )

        logger.info(
            f"Loading Whisper model: {model_cfg['name']} on {self.device}"
        )

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def validate_folder(
        self,
        folder: Path,
        wavs: List[Path],
        txts: List[Path],
        writer: csv.DictWriter,
    ):
        txt_map = {t.stem: t for t in txts}
        matched = [(w, txt_map[w.stem]) for w in wavs if w.stem in txt_map]

        if not matched:
            logger.warning(f"No matching audio/text pairs in {folder}")
            return

        ratio_diffs = []
        status = "PASS"
        for audio_file, text_file in matched:
            try:
                result = self.model.transcribe(str(audio_file))
                predicted_text = result["text"].strip()

                with text_file.open("r") as f:
                    reference_text = f.read().strip()

                pred_count = len(predicted_text.split())
                ref_count = len(reference_text.split())

                if ref_count == 0:
                    continue

                ratio_diff = abs(pred_count - ref_count) / ref_count
                ratio_diffs.append(ratio_diff)
                if ratio_diff > self.ratio_threshold:
                    status = "FAIL"
                    logger.debug(
                        f"Ratio diff too high for {audio_file.name}: {ratio_diff:.3f}"
                    )
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                status = "FAIL"
        avg_ratio_diff = (
            sum(ratio_diffs) / len(ratio_diffs) if ratio_diffs else 0.0
        )
        writer.writerow(
            {
                "folder": folder,
                "num_sentences": len(matched),
                "avg ratio diff": f"{avg_ratio_diff:.4f}",
                "status": status,
            }
        )

    def iter_folders(self, prefix: str = None):
        """Yield (dirpath, wav_files, txt_files) tuples where both exist."""
        for dirpath, _, files in os.walk(self.root_dir):
            dirpath = Path(dirpath)
            if prefix and not dirpath.name.startswith(prefix):
                continue

            wavs = sorted(
                [dirpath / f for f in files if f.lower().endswith(".wav")]
            )
            txts = sorted(
                [dirpath / f for f in files if f.lower().endswith(".txt")]
            )

            if wavs and txts:
                yield dirpath, wavs, txts

    def open_csv_writer(self, csv_name: str):
        """Open CSV and return a tuple (file_handle, writer)."""
        csv_path = self.root_dir / csv_name
        f = csv_path.open("w", encoding="utf-8", newline="")
        fieldnames = ["folder", "num_sentences", "avg_ratio_diff", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        return f, writer, csv_path

    def validate(self) -> None:
        """Validate folders recursively. If prefix is given, only process matching folders."""
        csv_name = (
            f"{self.folder_prefix}_{self.val_csv_output}"
            if self.folder_prefix
            else self.val_csv_output
        )
        csv_file, writer, csv_path = self.open_csv_writer(csv_name)
        logger.info(
            f"Starting validation under {self.root_dir} (prefix={self.folder_prefix})"
        )

        matched = False
        with csv_file:
            for dirpath, wavs, txts in self.iter_folders(
                prefix=self.folder_prefix
            ):
                matched = True
                logger.info(f"Validating folder: {dirpath}")
                self.validate_folder(dirpath, wavs, txts, writer)

        if not matched:
            logger.warning(
                f"No folders found matching prefix '{self.folder_prefix}'"
            )
        else:
            logger.success(f"Validation finished. Results saved to: {csv_path}")


if __name__ == "__main__":
    validator = ValidateSplit(config_path="config.yaml")
    validator.validate()
