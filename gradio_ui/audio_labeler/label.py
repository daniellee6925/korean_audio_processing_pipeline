import json
import uuid
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import gradio as gr
import click
import pandas as pd
from loguru import logger


DEFAULT_ADMIN_USERNAME = "boson"
DEFAULT_ADMIN_PASSWORD = "b"

DEFAULT_LABELER_CREDENTIALS = {
    "labeler1": "Label1!",
    "labeler2": "Label2!",
    "labeler3": "Label3!",
    "labeler4": "Label4!",
    "labeler5": "Label5!",
}


@dataclass
class AudioFile:
    """Represents a single audio file."""

    download_id: str
    segment_id: str
    download_path: Path
    audio_path: Path
    transcript: str
    start_end: list[float]
    duration: float

    @classmethod
    def from_segment_idx(
        cls, segment_idx: int, download_path: Path, audio_path: Path, csv_path: Path
    ):
        df = pd.read_csv(csv_path)
        row = df[df["segment_file"].str.contains(f"segment_{segment_idx}")].iloc[0]

        start_sec = row["start_sec"]
        end_sec = row["end_sec"]
        duration_sec = row["duration_sec"]

        transcript = ""
        segment_txt_path = download_path / f"segment_{segment_idx}.txt"
        if segment_txt_path.exists():
            with open(segment_txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()

        try:
            download_id = int(download_path.stem.split("_")[-2])
        except Exception:
            download_id = download_path.stem

        return cls(
            download_id=download_id,
            segment_id=segment_idx,
            download_path=download_path,
            audio_path=audio_path,
            transcript=transcript,
            start_end=[start_sec, end_sec],
            duration=duration_sec,
        )

    @classmethod
    def from_root_dir(cls, root_dir: Path) -> List["AudioFile"]:
        all_segments: List["AudioFile"] = []
        for audio_path in root_dir.rglob("*.wav"):
            donwload_path = audio_path.parent
            csv_path = donwload_path / "segment_all.csv"
            segment_idx = int(audio_path.stem.split("_")[-1])
            all_segments.append(
                cls.from_segment_idx(segment_idx, donwload_path, audio_path, csv_path)
            )

        return all_segments


@dataclass
class QualityAnnotation:
    """Represents an annotation for an audio bundle"""

    language_code_switch: bool = False
    domain_words: bool = False
    excessive_fillers: bool = False
    dysfluency: bool = False
    bad_audio_quality: bool = False
    stuttering: bool = False
    bad_pronounciation: bool = False


@dataclass
class HallucinationAnnotation:
    hallucination_level: str = "none"  # none, minor, major
    replace: bool = False  # model replaces word with similar sounding word
    remove: bool = False  # model doesn't hear certain sounds
    add: bool = False  # model adds certain words
    inferential: bool = False  # model falsely associates a sound or word not present in the audio
    injection_from_inaudible: bool = False  # hallucination from inaudible or short filler words


@dataclass
class Annotation:
    """Represents an annotation for an audio bundle"""

    user_id: str
    download_path: str
    audio_path: str
    asr_model: str
    transcript: str
    manual_transcript: str
    start_end: list[float]
    duration: float
    quality: QualityAnnotation
    hallucination: HallucinationAnnotation
    notes: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        # Ensure paths are strings
        if isinstance(result.get("download_path"), Path):
            result["download_path"] = str(result["download_path"])
        if isinstance(result.get("audio_path"), Path):
            result["audio_path"] = str(result["audio_path"])
        return asdict(self)


@dataclass
class PartitionConfig:
    """Configuration for partitioning files among labelers."""

    enabled: bool = False
    num_partitions: int = 5  # default to 5 partitions

    def get_partition_for_file(self, file_id: str, partition_idx: int) -> bool:
        """Determine if a file belongs to a specific partition."""
        if not self.enabled:
            return True

        hash_value = int(hashlib.md5(file_id.encode()).hexdigest(), 16)
        return (hash_value % self.num_partitions) == partition_idx

    def get_user_partition_idx(self, user_id: str) -> int:
        """Get partition index for a user based on id"""
        user_partition_map = {
            "labeler1": 0,
            "labeler2": 1,
            "labeler3": 2,
            "labeler4": 3,
            "labeler5": 4,
            "labeler6": 5,
            "labeler7": 6,
            "labeler8": 7,
            "labeler9": 8,
            "labeler10": 9,
        }
        if user_id in user_partition_map:
            return user_partition_map[user_id] % self.num_partitions
        # Fallback: hash user_id to assign partition
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return hash_value % self.num_partitions


class AnnotationManager:
    """Manages loading files and saving annotations."""

    def __init__(
        self,
        root_dir: str,
        anno_dir: str,
        partition_config: Optional[PartitionConfig] = None,
        model: str = "whisper-v3-large",
    ):
        self.root_dir = Path(root_dir)
        self.anno_dir = Path(anno_dir)
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        self.files: List[AudioFile] = []
        self.partition_config = partition_config or PartitionConfig()
        self.audio_file_map = {}
        self._load_files()
        self._print_partition_stats()
        self.model = model

    def _load_files(self):
        try:
            self.files = AudioFile.from_root_dir(self.root_dir)
            self.audio_file_map = {str(f.audio_path): f for f in self.files}
        except Exception as e:
            logger.error(f"Error loading {self.root_dir}: {e}")

        random.shuffle(self.files)

        logger.info(f"Loaded and randomized {len(self.files)} files")

    def _print_partition_stats(self):
        """Print statistics about how Files are distributed across partitions."""
        if not self.partition_config.enabled:
            return
        partition_counts = [0] * self.partition_config.num_partitions

        for f in self.files:
            hash_value = int(hashlib.md5(str(f.audio_path).encode()).hexdigest(), 16)
            partition_idx = hash_value % self.partition_config.num_partitions
            partition_counts[partition_idx] += 1

        for i, count in enumerate(partition_counts):
            logger.info(f"  Partition {i}: {count} files")

        all_annotated_ids = self.get_all_anno_file_ids()
        logger.info(f"\nAnnotation Status:")
        logger.info(f"  Total annotated: {len(all_annotated_ids)} files")
        logger.info(f"  Remaining: {len(self.files) - len(all_annotated_ids)} files")

        # Verify no overlap between partitions
        self._verify_no_partition_overlap()

    def _verify_no_partition_overlap(self):
        """Verify that files are uniquely assigned to partitions with no overlap."""
        if not self.partition_config.enabled:
            return

        file_assignments = {}

        for f in self.files:
            assigned_partitions = []
            for partition_idx in range(self.partition_config.num_partitions):
                if self.partition_config.get_partition_for_file(str(f.audio_path), partition_idx):
                    assigned_partitions.append(partition_idx)
            if len(assigned_partitions) == 0:
                logger.warning(
                    f"WARNING: Bundle {str(f.audio_path)} is not assigned to any partition!"
                )
            elif len(assigned_partitions) > 1:
                logger.error(
                    f"ERROR: Bundle {str(f.audio_path)} is assigned to multiple partitions: {assigned_partitions}"
                )
            else:
                file_assignments[str(f.audio_path)] = assigned_partitions[0]
        if len(file_assignments) == len(self.files):
            logger.info(
                f"\nPartition verification: OK - All {len(self.files)} files uniquely assigned"
            )
        else:
            logger.warning(
                f"\nPartition verification: FAILED - {len(self.files) - len(file_assignments)} files have assignment issues"
            )

    def get_user_files(self, user_id: str) -> List[AudioFile]:
        if not self.partition_config.enabled:
            return self.files

        partition_idx = self.partition_config.get_user_partition_idx(user_id)
        return [
            f
            for f in self.files
            if self.partition_config.get_partition_for_file(str(f.audio_path), partition_idx)
        ]

    def get_all_anno_file_ids(self) -> set:
        """Get all file paths that has been annotated by any user."""
        annotated_ids = set()

        if self.anno_dir.exists():
            for user_dir in self.anno_dir.iterdir():
                if user_dir.is_dir():
                    for ann_file in user_dir.rglob("*.json"):
                        try:
                            with open(ann_file, "r", encoding="utf-8") as f:
                                ann_data = json.load(f)
                                file_path = ann_data.get("audio_path")
                                if file_path:
                                    annotated_ids.add(file_path)
                        except Exception:
                            continue
        return annotated_ids

    def get_all_unanno_files(self, user_id: str) -> List[AudioFile]:
        """Get files that haven't been annotated by ANY user, respecting partitions"""
        user_files = self.get_user_files(user_id)
        all_annotated_ids = self.get_all_anno_file_ids()

        return [f for f in user_files if str(f.audio_path) not in all_annotated_ids]

    @staticmethod
    def get_quality_annotation(quality_desc: list[str] = []):
        qannotation = QualityAnnotation(
            language_code_switch="language_code_switch" in quality_desc,
            domain_words="domain_words" in quality_desc,
            excessive_fillers="excessive_fillers" in quality_desc,
            dysfluency="dysfluency" in quality_desc,
            bad_audio_quality="bad_audio_quality" in quality_desc,
            stuttering="stuttering" in quality_desc,
            bad_pronounciation="bad_pronounciation" in quality_desc,
        )
        return qannotation

    @staticmethod
    def get_hallucination_annotation(
        hallucination_level: str = "none",
        hallucinations_desc: list[str] = [],
    ):
        hannotation = HallucinationAnnotation(
            hallucination_level=hallucination_level,
            replace="replace" in hallucinations_desc,
            remove="remove" in hallucinations_desc,
            add="add" in hallucinations_desc,
            inferential="inferential" in hallucinations_desc,
            injection_from_inaudible="injection_from_inaudible" in hallucinations_desc,
        )
        return hannotation

    def save_annotation(
        self,
        user_id,
        audio_path,
        transcript,
        quality_desc,
        hallucination_radio,
        hallucinations_desc,
        notes,
    ):
        audio_path_str = str(audio_path) if isinstance(audio_path, Path) else audio_path

        f = self.audio_file_map[audio_path_str]
        qannotation = self.get_quality_annotation(quality_desc)
        hannotation = self.get_hallucination_annotation(hallucination_radio, hallucinations_desc)
        annotation = Annotation(
            user_id=user_id,
            download_path=str(f.download_path),
            audio_path=audio_path_str,
            asr_model=self.model,
            transcript=f.transcript,
            manual_transcript=transcript,
            start_end=f.start_end,
            duration=f.duration,
            quality=qannotation,
            hallucination=hannotation,
            notes=notes,
        )
        return annotation

    def save_json(self, annotation: Annotation):
        """Save an annotation json file to dir"""
        user_dir = self.anno_dir / annotation.user_id
        audio_path = Path(annotation.audio_path)

        parent_path = user_dir / audio_path.parent
        parent_path.mkdir(parents=True, exist_ok=True)

        filename = audio_path.name.replace(".wav", ".json")
        filepath = parent_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath

    def get_progress(self, user_id: str) -> Tuple[int, int]:
        """Get annotation progress for a user (completed, total)."""
        user_files = self.get_user_files(user_id)

        all_annotated_ids = self.get_all_anno_file_ids()
        completed = len([f for f in user_files if str(f.audio_path) in all_annotated_ids])
        return completed, len(user_files)


class AudioAnnotator:
    """Main class for Annotating data"""

    def __init__(
        self,
        root_dir: str,
        anno_dir: str,
        partition_config: Optional[PartitionConfig] = None,
        model: str = "whisper-v3-large",
    ):
        self.manager = AnnotationManager(root_dir, anno_dir, partition_config, model)
        self.partition_config = partition_config or PartitionConfig()

        self.user_credentials = {DEFAULT_ADMIN_USERNAME: DEFAULT_ADMIN_PASSWORD}
        self.user_credentials.update(DEFAULT_LABELER_CREDENTIALS)

        if self.partition_config.num_partitions > len(DEFAULT_LABELER_CREDENTIALS):
            for i in range(
                len(DEFAULT_LABELER_CREDENTIALS) + 1, self.partition_config.num_partitions + 1
            ):
                username = f"labeler{i}"
                password = f"Label{i}!"
                self.user_credentials[username] = password
        self.current_file_idx = {}

        # Path to store discarded files
        self.discard_file_path = Path(anno_dir) / "discarded_files.json"
        self.discarded_files = self._load_discarded_files()

    def _load_discarded_files(self) -> dict:
        """Load discarded files from disk"""
        if self.discard_file_path.exists():
            try:
                with open(self.discard_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading discarded files: {e}")
                return {}
        return {}

    def _save_discarded_files(self):
        """Save discarded files to disk"""
        try:
            with open(self.discard_file_path, "w", encoding="utf-8") as f:
                json.dump(self.discarded_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving discarded files: {e}")

    def authenticate(self, username: str, password: str) -> Tuple[str, gr.update, gr.update, str]:
        """Authenticate login"""
        if username in self.user_credentials and self.user_credentials[username] == password:
            self.current_file_idx[username] = 0

            # Initialize discard list for user if not exists
            if username not in self.discarded_files:
                self.discarded_files[username] = []

            login_msg = f"**Logged in as `{username}` **"
            if self.partition_config.enabled and username != DEFAULT_ADMIN_USERNAME:
                partition_idx = self.partition_config.get_user_partition_idx(username)
                user_files = self.manager.get_user_files(username)
                login_msg += f"\nAssigned to partition {partition_idx} ({len(user_files)} files)"

            # Show discarded count if any
            discarded_count = len(self.discarded_files.get(username, []))
            if discarded_count > 0:
                login_msg += f"\n{discarded_count} files previously discarded"

            return (
                login_msg,
                gr.update(visible=False),  # hide login
                gr.update(visible=True),  # show main
                username,
            )
        return (
            "Invalid username or password",
            gr.update(visible=True),
            gr.update(visible=False),
            None,
        )

    def discard_file(
        self,
        user_id: str,
        audio_path: str,
    ):
        """Discard the current file and load next one"""
        if not user_id or not audio_path:
            return (
                None,
                "",
                "Progress: 0/0",
                None,
                "Error: Missing user or file information",
                "",
            )

        # Add to discarded stack
        if user_id not in self.discarded_files:
            self.discarded_files[user_id] = []
        self.discarded_files[user_id].append(audio_path)

        # Save to disk
        self._save_discarded_files()

        result = self.load_next_file(user_id)
        return result[:4] + (f"File discarded and saved. You can undo this action.",) + (result[5],)

    def undo_discard(self, user_id: str):
        """Undo the last discarded file and load it"""
        if not user_id:
            return (
                None,
                "",
                "Progress: 0/0",
                None,
                "Error: Please login first",
                "",
            )

        if user_id not in self.discarded_files or not self.discarded_files[user_id]:
            # No discarded files, just reload current state
            result = self.load_next_file(user_id)
            return result[:4] + ("No discarded files to undo.",) + (result[5],)

        # Pop the last discarded file
        restored_path = self.discarded_files[user_id].pop()

        # Save to disk
        self._save_discarded_files()

        # Find the file in our files list
        restored_file = self.manager.audio_file_map.get(restored_path)

        if not restored_file:
            result = self.load_next_file(user_id)
            return result[:4] + ("Error: Could not find discarded file.",) + (result[5],)

        completed, total = self.manager.get_progress(user_id)
        discarded_count = len(self.discarded_files.get(user_id, []))
        unannotated = self.manager.get_all_unanno_files(user_id)
        discarded_paths = set(self.discarded_files.get(user_id, []))
        remaining = len([f for f in unannotated if str(f.audio_path) not in discarded_paths])

        progress_text = f"Progress: {completed}/{total} completed | {discarded_count} discarded | {remaining} remaining"
        file_path_display = f"**Current file:** `{str(restored_file.audio_path)}`"

        return (
            str(restored_file.audio_path),  # audio_player
            restored_file.transcript,  # transcription_box
            progress_text,  # progress_text
            str(restored_file.audio_path),  # audio_path_state
            "Restored previously discarded file.",  # status_message
            file_path_display,  # file_path_display
        )

    def load_next_file(self, user_id: str) -> Tuple:
        """Load the next unannotated file for the user"""
        if not user_id:
            return (None, "", "Please login first", None, "", "")
        unannotated = self.manager.get_all_unanno_files(user_id)

        # Filter out discarded files
        discarded_paths = set(self.discarded_files.get(user_id, []))
        unannotated = [f for f in unannotated if str(f.audio_path) not in discarded_paths]

        if not unannotated:
            return (None, "", "All bundles annotated or discarded!", None, "", "")
        cur_file = random.choice(unannotated)

        completed, total = self.manager.get_progress(user_id)
        discarded_count = len(self.discarded_files.get(user_id, []))
        progress_text = f"Progress: {completed}/{total} completed | {discarded_count} discarded | {len(unannotated)} remaining"

        # Format the file path display
        file_path_display = f"**Current file:** `{str(cur_file.audio_path)}`"

        return (
            str(cur_file.audio_path),  # audio_player
            cur_file.transcript,  # transcription_box
            progress_text,  # progress_text
            str(cur_file.audio_path),  # audio_path_state
            "",  # status_message - Clear status message
            file_path_display,  # file_path_display
        )

    def submit_annotation(
        self,
        user_id: str,
        audio_path: str,
        transcription_box: str,
        quality_checkbox: list[str],
        hallucination_radio: str,
        hallucination_checkbox: list[str],
        notes: str = "",
    ):
        if not user_id or not audio_path:
            return (
                None,
                "",
                "Progress: 0/0",
                None,
                "Error: Missing user or file information",
                "",
            )

        annotation = self.manager.save_annotation(
            user_id,
            audio_path,
            transcription_box,
            quality_checkbox,
            hallucination_radio,
            hallucination_checkbox,
            notes,
        )
        self.manager.save_json(annotation)
        result = self.load_next_file(user_id)
        return result[:4] + ("Annotation saved!",) + (result[5],)

    def create_interface(self) -> gr.Blocks:
        """Create gradio interface"""
        quality_fields = [
            "language_code_switch",
            "domain_words",
            "excessive_fillers",
            "dysfluency",
            "bad_audio_quality",
            "stuttering",
            "bad_pronounciation",
        ]
        hallucination_fields = [
            "replace",
            "remove",
            "add",
            "inferential",
            "injection_from_inaudible",
        ]
        with gr.Blocks(title="Audio Annotation Platform", theme="soft") as interface:
            user_state = gr.State(None)
            audio_path_state = gr.State(None)

            gr.Markdown("# Audio Annotation Platform")

            if self.partition_config.enabled:
                gr.Markdown(
                    f"""
                ### Partition Configuration
                - **Mode**: {self.partition_config.enabled}
                - **Number of partitions**: {self.partition_config.num_partitions}
                - **Available users**: {", ".join([f"{u} (pwd: {p})" for u, p in self.user_credentials.items() if u != DEFAULT_ADMIN_USERNAME])}
                """
                )
            # login page
            with gr.Row(visible=True) as login_page:
                with gr.Column(scale=1):
                    gr.Markdown("### Login")
                    username_input = gr.Textbox(label="Username", placeholder="enter username")
                    password_input = gr.Textbox(label="Password", placeholder="enter password")
                    login_button = gr.Button("Login", variant="primary")
                    login_message = gr.Markdown()
            # main section
            with gr.Column(visible=False) as main_page:
                progress_text = gr.Markdown("Progress 0/0")
                status_message = gr.Markdown("")  # Status message for actions
                file_path_display = gr.Markdown("")  # Display current file path

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Audio File")
                        audio_player = gr.Audio(
                            label="Audio Player", interactive=False, autoplay=True
                        )
                        transcription_box = gr.Textbox(
                            label="Model Transcript", interactive=True, lines=3, max_lines=5
                        )

                # Annotation Controls
                with gr.Column(visible=True) as annotation_controls:
                    gr.Markdown("### Audio Quality Evaluation")
                    quality_checkbox = gr.CheckboxGroup(
                        label="Quality Annotations",
                        choices=quality_fields,
                        value=[],
                    )
                    hallucination_radio = gr.Radio(
                        label="Hallucination Level",
                        choices=["none", "minor", "major"],
                        value="none",
                    )

                    hallucination_checkbox = gr.CheckboxGroup(
                        label="Hallucination Flags", choices=hallucination_fields, value=[]
                    )
                    notes_input = gr.Textbox(
                        label="Additional Notes (optional)",
                        interactive=True,
                        lines=2,
                        placeholder="Add any additional observations...",
                    )

                with gr.Row():
                    undo_button = gr.Button("↩️ Undo Discard", variant="secondary", size="lg")
                    discard_button = gr.Button("❌ Discard", variant="stop", size="lg")
                    submit_button = gr.Button("✅ Keep (Next)", variant="primary", size="lg")

            # event handlers
            login_button.click(
                self.authenticate,
                inputs=[username_input, password_input],
                outputs=[login_message, login_page, main_page, user_state],
            ).then(
                self.load_next_file,
                inputs=[user_state],
                outputs=[
                    audio_player,
                    transcription_box,
                    progress_text,
                    audio_path_state,
                    status_message,
                    file_path_display,
                ],
            )

            submit_button.click(
                self.submit_annotation,
                inputs=[
                    user_state,
                    audio_path_state,
                    transcription_box,
                    quality_checkbox,
                    hallucination_radio,
                    hallucination_checkbox,
                    notes_input,
                ],
                outputs=[
                    audio_player,
                    transcription_box,
                    progress_text,
                    audio_path_state,
                    status_message,
                    file_path_display,
                ],
            ).then(
                lambda: ([], "none", [], ""),
                inputs=None,
                outputs=[
                    quality_checkbox,
                    hallucination_radio,
                    hallucination_checkbox,
                    notes_input,
                ],
            )

            discard_button.click(
                self.discard_file,
                inputs=[user_state, audio_path_state],
                outputs=[
                    audio_player,
                    transcription_box,
                    progress_text,
                    audio_path_state,
                    status_message,
                    file_path_display,
                ],
            ).then(
                lambda: ([], "none", [], ""),
                inputs=None,
                outputs=[
                    quality_checkbox,
                    hallucination_radio,
                    hallucination_checkbox,
                    notes_input,
                ],
            )

            undo_button.click(
                self.undo_discard,
                inputs=[user_state],
                outputs=[
                    audio_player,
                    transcription_box,
                    progress_text,
                    audio_path_state,
                    status_message,
                    file_path_display,
                ],
            ).then(
                lambda: ([], "none", [], ""),
                inputs=None,
                outputs=[
                    quality_checkbox,
                    hallucination_radio,
                    hallucination_checkbox,
                    notes_input,
                ],
            )

        return interface


@click.command()
@click.option(
    "--audio-dir",
    default="podbbang",
    help="Directory containing audio bundles",
)
@click.option(
    "--annotations-dir",
    default="gradio_ui/audio_labeler/annotations",
    help="Directory to save annotations",
)
@click.option(
    "--model",
    default="whisper-v3-large",
    help="Model used to transcribe",
)
@click.option("--port", default=7680, help="Port to run the server on")
@click.option("--share", is_flag=True, help="Create a public shareable link")
@click.option(
    "--enable-partitions",
    is_flag=True,
    default=False,
    help="Enable automatic partitioning of bundles among labelers (default: enabled)",
)
@click.option("--num-partitions", default=5, help="Number of partitions to divide bundles into")
@click.option(
    "--partition-mode",
    type=click.Choice(["hash", "sequential"]),
    default="hash",
    help="Partitioning mode: hash (deterministic) or sequential (round-robin)",
)
def main(
    audio_dir,
    annotations_dir,
    model,
    port,
    share,
    enable_partitions,
    num_partitions,
    partition_mode,
):
    """Launch the Annotation Platform Gradio interface with optional partitioning."""
    partition_config = PartitionConfig(
        enabled=enable_partitions,
        num_partitions=num_partitions,
    )

    labeler = AudioAnnotator(audio_dir, annotations_dir, partition_config, model)

    interface = labeler.create_interface()

    logger.info(f"Launching Audio Annotation Platform...")
    logger.info(f"Audio files directory: {audio_dir}")
    logger.info(f"Annotations directory: {annotations_dir}")

    if enable_partitions:
        logger.info(f"\nPartitioning enabled:")
        logger.info(f"  - Mode: {partition_mode}")
        logger.info(f"  - Number of partitions: {num_partitions}")
        logger.info(f"  - Users created: labeler1 to labeler{num_partitions}")
        logger.info(f"\nUser credentials:")
        for username, password in labeler.user_credentials.items():
            if username != DEFAULT_ADMIN_USERNAME:  # Don't show admin password in logs
                logger.info(f"  - {username}: {password}")

    interface.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    main()
