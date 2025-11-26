import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger
from tqdm import tqdm


class AcousticDifficultyFilter:
    """Filter transcripts based on acoustic/transcription difficulty."""

    def __init__(self, root_dir: str, keep_hard: bool = True):
        self.root_dir = Path(root_dir)
        self.keep_hard = keep_hard
        self.deleted_files = 0

        # Setup difficulty criteria
        self.setup_acoustic_difficulty_rules()

    def setup_acoustic_difficulty_rules(self):
        """Define what makes audio acoustically challenging to transcribe."""

        # 1. Phonetically similar/confusing Korean sounds
        self.confusing_pairs = [
            ("ㄱ", "ㅋ"),
            ("ㄷ", "ㅌ"),
            ("ㅂ", "ㅍ"),  # Aspirated vs unaspirated
            ("ㄴ", "ㅇ"),
            ("ㄹ", "ㄴ"),  # Similar nasals
            ("ㅐ", "ㅔ"),
            ("ㅒ", "ㅖ"),  # Similar vowels
            ("ㅗ", "ㅜ"),
            ("ㅓ", "ㅕ"),  # Back vowels
        ]

        # 2. Homophones and near-homophones (sound same but different meaning)
        self.homophones = {
            "가다",
            "갔다",  # Tense differences
            "있다",
            "없다",  # Presence/absence
            "되다",
            "돼다",
            "뒤다",
            "안",
            "않",  # Negation confusion
            "의",
            "에",
            "이",  # Particles
        }

        # 3. Words with double consonants (hard to detect)
        self.double_consonant_pattern = r"[ㄱ-ㅎ]{2,}"

        # 4. Rapid speech markers (contracted forms)
        self.contractions = [
            "거예요",
            "거야",  # 것이에요 -> 거예요
            "뭐야",
            "뭐예요",  # 무엇
            "그래",
            "그럼",  # 그러면
            "됐어",
            "됐다",  # Rapid past tense
        ]

        # 5. Words with final consonants followed by initial consonants (liaison)
        self.liaison_pattern = r"[ㄱ-ㅎ][가-힣][ㄱ-ㅎ]"

        # 6. Numbers (hard to transcribe accurately)
        self.number_pattern = r"\d+|[일이삼사오육칠팔구십백천만억]+"

        # 7. English/foreign words in Korean (code-switching)
        self.english_pattern = r"[A-Za-z]{2,}"

        # 8. Very short utterances (high error rate due to lack of context)
        self.min_length_for_easy = 5
        self.max_length_for_hard = 3

        # 9. Repeated syllables (stuttering or emphasis - hard to transcribe correctly)
        self.repeated_syllable_pattern = r"(.)\1{2,}"

    def has_phonetic_confusion(self, text: str) -> bool:
        """Check if text contains phonetically confusing sound pairs."""
        for char1, char2 in self.confusing_pairs:
            if char1 in text and char2 in text:
                return True
        return False

    def has_homophones(self, text: str) -> bool:
        """Check if text contains common homophones."""
        return any(homo in text for homo in self.homophones)

    def has_double_consonants(self, text: str) -> bool:
        """Check for double consonants (gemination)."""
        # Korean words with ㄲ, ㄸ, ㅃ, ㅆ, ㅉ
        double_chars = ["ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"]
        return any(dc in text for dc in double_chars)

    def has_contractions(self, text: str) -> bool:
        """Check for contracted/rapid speech forms."""
        return any(contraction in text for contraction in self.contractions)

    def has_numbers(self, text: str) -> bool:
        """Check if text contains numbers (numeric or Korean)."""
        return bool(re.search(self.number_pattern, text))

    def has_code_switching(self, text: str) -> bool:
        """Check for English/foreign words mixed with Korean."""
        return bool(re.search(self.english_pattern, text))

    def is_very_short(self, text: str) -> bool:
        """Very short utterances lack context and are error-prone."""
        return len(text.strip()) <= self.max_length_for_hard

    def has_repeated_syllables(self, text: str) -> bool:
        """Check for repeated syllables (stuttering, emphasis)."""
        return bool(re.search(self.repeated_syllable_pattern, text))

    def count_particle_density(self, text: str) -> float:
        """
        High particle density = more grammar complexity = harder transcription.
        Korean particles: 은/는, 이/가, 을/를, 에, 에서, 으로, 와/과, etc.
        """
        particles = [
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "에서",
            "으로",
            "와",
            "과",
            "의",
            "도",
            "만",
            "부터",
            "까지",
        ]
        particle_count = sum(text.count(p) for p in particles)

        words = len(text.split())
        if words == 0:
            return 0

        return particle_count / words

    def has_similar_sounding_words_nearby(self, text: str) -> bool:
        """
        Check if similar-sounding words appear in same sentence.
        This causes acoustic confusion for models.
        """
        words = text.split()

        # Check for repeated similar sounds
        for i in range(len(words) - 1):
            # Simple similarity: first 2 characters match
            if len(words[i]) >= 2 and len(words[i + 1]) >= 2:
                if words[i][:2] == words[i + 1][:2]:
                    return True

        return False

    def is_acoustically_hard(self, text: str) -> bool:
        """
        Determine if a transcript would be acoustically hard to transcribe.
        Returns True if the audio/transcript has characteristics that challenge ASR models.
        """
        text = text.strip()

        if not text:
            return False

        hard_score = 0

        # Criterion 1: Very short (lacks context) - STRONG indicator
        if self.is_very_short(text):
            hard_score += 3

        # Criterion 2: Contains phonetically confusing sounds
        if self.has_phonetic_confusion(text):
            hard_score += 2

        # Criterion 3: Contains homophones
        if self.has_homophones(text):
            hard_score += 2

        # Criterion 4: Has double consonants (hard to detect acoustically)
        if self.has_double_consonants(text):
            hard_score += 1

        # Criterion 5: Contains contractions (rapid speech)
        if self.has_contractions(text):
            hard_score += 2

        # Criterion 6: Contains numbers (high WER)
        if self.has_numbers(text):
            hard_score += 2

        # Criterion 7: Code-switching (English + Korean)
        if self.has_code_switching(text):
            hard_score += 2

        # Criterion 8: Repeated syllables
        if self.has_repeated_syllables(text):
            hard_score += 2

        # Criterion 9: High particle density (complex grammar)
        if self.count_particle_density(text) > 0.3:
            hard_score += 1

        # Criterion 10: Similar sounding words nearby
        if self.has_similar_sounding_words_nearby(text):
            hard_score += 2

        # Threshold: score >= 3 is considered "hard"
        return hard_score >= 7

    def should_keep_file(self, txt_path: Path) -> bool:
        """Decide if a transcript file should be kept based on acoustic difficulty."""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            is_hard = self.is_acoustically_hard(text)

            # Keep hard if keep_hard=True, keep easy if keep_hard=False
            return is_hard if self.keep_hard else not is_hard

        except Exception as e:
            logger.error(f"Error reading {txt_path}: {e}")
            return True  # Keep if error (safe default)

    def get_audio_path(self, txt_path: Path) -> Path:
        """Convert w_segment_3.txt -> segment_3.wav"""
        txt_name = txt_path.stem  # e.g., "w_segment_3"

        # Remove 'w_' prefix if present
        if txt_name.startswith("w_"):
            audio_name = txt_name[2:] + ".wav"  # "segment_3.wav"
        else:
            audio_name = txt_name + ".wav"

        audio_path = txt_path.parent / audio_name
        return audio_path

    def find_all_transcripts(self) -> List[Path]:
        """Find all .txt files in directory tree."""
        txt_files = list(self.root_dir.rglob("*.txt"))
        logger.info(f"Found {len(txt_files)} transcript files")
        return txt_files

    def delete_file_pair(self, txt_path: Path) -> Tuple[bool, bool]:
        """Delete both transcript and audio file. Returns (txt_deleted, audio_deleted)."""
        txt_deleted = False
        audio_deleted = False

        # Delete transcript
        try:
            txt_path.unlink()
            txt_deleted = True
            self.deleted_files += 1
        except Exception as e:
            logger.error(f"Failed to delete {txt_path}: {e}")

        # Delete corresponding audio
        audio_path = self.get_audio_path(txt_path)
        if audio_path.exists():
            try:
                audio_path.unlink()
                audio_deleted = True
                self.deleted_files += 1
            except Exception as e:
                logger.error(f"Failed to delete {audio_path}: {e}")
        else:
            logger.warning(f"Audio file not found: {audio_path}")

        return txt_deleted, audio_deleted

    def analyze_sample(self, txt_path: Path) -> Dict:
        """Analyze a single transcript and return difficulty breakdown."""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            analysis = {
                "path": str(txt_path),
                "text": text,
                "length": len(text),
                "very_short": self.is_very_short(text),
                "phonetic_confusion": self.has_phonetic_confusion(text),
                "homophones": self.has_homophones(text),
                "double_consonants": self.has_double_consonants(text),
                "contractions": self.has_contractions(text),
                "numbers": self.has_numbers(text),
                "code_switching": self.has_code_switching(text),
                "repeated_syllables": self.has_repeated_syllables(text),
                "particle_density": self.count_particle_density(text),
                "similar_sounds": self.has_similar_sounding_words_nearby(text),
                "is_hard": self.is_acoustically_hard(text),
            }

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing {txt_path}: {e}")
            return None

    def show_samples(self, n: int = 10):
        """Show sample analysis of n files (for debugging/tuning)."""
        txt_files = self.find_all_transcripts()

        import random

        samples = random.sample(txt_files, min(n, len(txt_files)))

        logger.info(f"\n{'='*80}")
        logger.info(f"SAMPLE ANALYSIS (showing {len(samples)} random files)")
        logger.info(f"{'='*80}\n")

        for txt_path in samples:
            analysis = self.analyze_sample(txt_path)
            if analysis:
                print(f"\nFile: {txt_path.name}")
                print(f"Text: {analysis['text'][:100]}")
                print(f"Length: {analysis['length']}")
                print(f"Hard: {analysis['is_hard']}")
                print(f"  - Very short: {analysis['very_short']}")
                print(f"  - Phonetic confusion: {analysis['phonetic_confusion']}")
                print(f"  - Homophones: {analysis['homophones']}")
                print(f"  - Double consonants: {analysis['double_consonants']}")
                print(f"  - Contractions: {analysis['contractions']}")
                print(f"  - Numbers: {analysis['numbers']}")
                print(f"  - Code-switching: {analysis['code_switching']}")
                print(f"  - Repeated syllables: {analysis['repeated_syllables']}")
                print(f"  - Particle density: {analysis['particle_density']:.2f}")
                print(f"  - Similar sounds nearby: {analysis['similar_sounds']}")
                print("-" * 80)

    def process_all(self, dry_run: bool = False):
        """Process all transcripts and remove easy/hard files based on acoustic difficulty."""
        txt_files = self.find_all_transcripts()

        if not txt_files:
            logger.error(f"No transcript files found in {self.root_dir}")
            return

        files_to_delete = []
        files_to_keep = []

        # First pass: determine what to keep/delete
        logger.info("Analyzing acoustic difficulty of transcripts...")
        for txt_path in tqdm(txt_files, desc="Analyzing"):
            if self.should_keep_file(txt_path):
                files_to_keep.append(txt_path)
            else:
                files_to_delete.append(txt_path)

        difficulty = "acoustically hard" if self.keep_hard else "acoustically easy"
        logger.info(f"Keeping {len(files_to_keep)} {difficulty} transcripts")
        logger.info(
            f"Deleting {len(files_to_delete)} {'easy' if self.keep_hard else 'hard'} transcripts"
        )

        if dry_run:
            logger.info("DRY RUN - No files deleted")
            logger.info(f"Would delete {len(files_to_delete) * 2} files (txt + audio)")

            # Show some examples of what would be deleted
            logger.info("\nSample files that would be DELETED:")
            for txt_path in files_to_delete[:5]:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                logger.info(f"  {txt_path.name}: {text[:60]}")

            logger.info("\nSample files that would be KEPT:")
            for txt_path in files_to_keep[:5]:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                logger.info(f"  {txt_path.name}: {text[:60]}")

            return

        # Second pass: delete files
        logger.info("Deleting files...")
        deleted_count = 0

        for txt_path in tqdm(files_to_delete, desc="Deleting"):
            txt_del, audio_del = self.delete_file_pair(txt_path)
            if txt_del or audio_del:
                deleted_count += 1

        logger.success(f"Deleted {self.deleted_files} files total")
        logger.success(f"Kept {len(files_to_keep)} {difficulty} transcript pairs")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter Korean transcripts by acoustic/transcription difficulty"
    )
    parser.add_argument("--root_dir", type=str, help="Root directory containing transcripts")
    parser.add_argument(
        "--keep-easy", action="store_true", help="Keep easy-to-transcribe sentences instead of hard"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting"
    )
    parser.add_argument(
        "--show-samples", type=int, metavar="N", help="Show N sample analyses without processing"
    )

    args = parser.parse_args()

    filter_obj = AcousticDifficultyFilter(root_dir=args.root_dir, keep_hard=not args.keep_easy)

    if args.show_samples:
        filter_obj.show_samples(n=args.show_samples)
        return

    logger.info(f"Root directory: {args.root_dir}")
    logger.info(f"Mode: Keep {'EASY' if args.keep_easy else 'HARD'} (acoustically) sentences")
    logger.info(f"Dry run: {args.dry_run}")

    filter_obj.process_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
