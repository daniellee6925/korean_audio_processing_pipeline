"""
Audio Age, Gender, and Traits Labeling using Google Gemini API

This script uses Google's Gemini API to analyze audio files and predict
the speaker's age range, gender, and voice traits, then updates corresponding JSON files.

Includes audio trimming to process only the first N seconds for faster processing.
"""

from dotenv import load_dotenv
import google.generativeai as genai
import os
from pathlib import Path
import json
from typing import Dict, Optional, List
from pydub import AudioSegment
import tempfile

load_dotenv()


class AudioLabeler:
    def __init__(self, api_key: str, max_duration_seconds: int = 10):
        """
        Initialize the Audio Labeler with Gemini API.

        Args:
            api_key: Your Google AI Studio API key
            max_duration_seconds: Maximum duration of audio to process (default: 10 seconds)
                                 Set to None to process entire audio file
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.max_duration_seconds = max_duration_seconds

    def _trim_audio(self, audio_path: str) -> str:
        """
        Trim audio to max_duration_seconds if needed.

        Args:
            audio_path: Path to the original audio file

        Returns:
            Path to trimmed audio file (or original if no trimming needed)
        """
        if self.max_duration_seconds is None:
            return audio_path

        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)

            # Check if trimming is needed
            duration_ms = len(audio)
            max_duration_ms = self.max_duration_seconds * 1000

            if duration_ms <= max_duration_ms:
                # No trimming needed
                return audio_path

            # Trim audio to max_duration_seconds
            trimmed_audio = audio[:max_duration_ms]

            # Create temporary file with same extension
            audio_ext = Path(audio_path).suffix
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext)
            temp_path = temp_file.name
            temp_file.close()

            # Export trimmed audio
            trimmed_audio.export(temp_path, format=audio_ext[1:])  # Remove the dot from extension

            print(f"  Trimmed audio from {duration_ms/1000:.1f}s to {self.max_duration_seconds}s")
            return temp_path

        except Exception as e:
            print(f"  Warning: Could not trim audio ({str(e)}), using original file")
            return audio_path

    def analyze_audio(self, audio_path: str, update_json: bool = True) -> Dict[str, str]:
        """
        Analyze an audio file to predict age, gender, and voice traits.

        Args:
            audio_path: Path to the audio file
            update_json: If True, update the corresponding JSON file with results

        Returns:
            Dictionary containing predicted age range, gender, and traits
        """
        temp_audio_path = None

        try:
            # Trim audio if max_duration is set
            processing_path = self._trim_audio(audio_path)
            if processing_path != audio_path:
                temp_audio_path = processing_path

            # Upload the audio file
            audio_file = genai.upload_file(path=processing_path)

            # Create a detailed prompt for age, gender, and traits detection
            prompt = """
            Please analyze this audio file and provide:
            1. The estimated age range of the speaker: 10s, 20s, 30s, 40s, 50s, 60s. DO NOT GIVE RANGES such as 20s-30s. Just pick one.
            2. The predicted gender of the speaker (male, female, or uncertain)
            3. Voice traits and characteristics (provide 3-6 descriptive traits)
            
            Base your analysis on vocal characteristics such as pitch, tone, speech patterns, delivery style, and emotional quality.
            
            For traits, consider characteristics like:
            - Tone: warm, cold, friendly, authoritative, casual, formal, professional
            - Delivery: smooth, rough, clear, raspy, breathy, nasal
            - Pace: fast, slow, measured, energetic, calm
            - Emotion: cheerful, serious, nervous, confident, enthusiastic, monotone
            - Character: intellectual, playful, mature, youthful, sophisticated
            - Style: narration, conversational, dramatic, informative, storytelling
            - Quality: soothing, commanding, gentle, powerful, soft, loud
            
            Respond in the following JSON format:
            {
                "age_range": "age range here (e.g., 20s, 30s)",
                "gender": "gender here (male/female/uncertain)",
                "traits": ["trait1", "trait2", "trait3", "trait4"],
                "confidence": "high/medium/low",
                "notes": "brief explanation of your analysis"
            }
            
            IMPORTANT: 
            - traits Must be an array of 2-6 lowercase descriptive words 
            - age_range MUST be a single decade (10s, 20s, 30s, 40s, 50s, or 60s)
            - Do not include markdown formatting in the JSON
            """

            # Generate response
            response = self.model.generate_content([prompt, audio_file])

            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.text
                # Remove markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                result = json.loads(response_text.strip())

                # Ensure traits is a list
                if "traits" not in result or not isinstance(result["traits"], list):
                    result["traits"] = []

                # Update the corresponding JSON file if requested
                if update_json:
                    self._update_json_file(audio_path, result)

                return result
            except (json.JSONDecodeError, IndexError) as e:
                # If parsing fails, return the raw response
                return {
                    "age_range": "unknown",
                    "gender": "unknown",
                    "traits": [],
                    "confidence": "low",
                    "notes": f"Could not parse response: {response.text}",
                }
        finally:
            # Clean up temporary trimmed audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

    def _update_json_file(self, audio_path: str, analysis_result: Dict) -> None:
        """
        Update the JSON file corresponding to the audio file with age, gender, and traits.

        Args:
            audio_path: Path to the audio file
            analysis_result: Analysis results containing age_range, gender, and traits
        """
        # Get the JSON file path (same name as audio file, but with .json extension)
        audio_file = Path(audio_path)
        json_path = audio_file.with_suffix(".json")

        try:
            # Read existing JSON file
            if json_path.exists():
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Update metadata fields
                if "metadata" not in data:
                    data["metadata"] = {}

                data["metadata"]["gender"] = analysis_result.get("gender", "unknown")
                data["metadata"]["age"] = analysis_result.get("age_range", "unknown")
                data["metadata"]["traits"] = analysis_result.get("traits", [])

                # Write back to JSON file
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"  Updated JSON: {json_path.name}")
            else:
                print(f"  Warning: JSON file not found: {json_path}")
        except Exception as e:
            print(f"  Error updating JSON file: {str(e)}")

    def batch_analyze(
        self, audio_directory: str, output_file: Optional[str] = None, update_json: bool = True
    ) -> Dict[str, Dict]:
        """
        Analyze multiple audio files in a directory.

        Args:
            audio_directory: Path to directory containing audio files
            output_file: Optional path to save results as JSON
            update_json: If True, update corresponding JSON files for each audio file

        Returns:
            Dictionary mapping filenames to their analysis results
        """
        results = {}
        audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

        audio_dir = Path(audio_directory)
        audio_files = list(audio_dir.rglob("*.wav"))

        print(f"Found {len(audio_files)} audio files to analyze...")
        if self.max_duration_seconds:
            print(
                f"Processing first {self.max_duration_seconds} seconds of each file for faster analysis\n"
            )

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\nProcessing {i}/{len(audio_files)}: {audio_file.name}")
            try:
                result = self.analyze_audio(str(audio_file), update_json=update_json)
                results[audio_file.name] = result
                print(f"  Age: {result.get('age_range', 'unknown')}")
                print(f"  Gender: {result.get('gender', 'unknown')}")
                print(f"  Traits: {', '.join(result.get('traits', []))}")
                print(f"  Confidence: {result.get('confidence', 'unknown')}")
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[audio_file.name] = {
                    "error": str(e),
                    "age_range": "error",
                    "gender": "error",
                    "traits": [],
                }

        # Save results if output file is specified
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")

        return results


def main():
    """
    Example usage of the AudioLabeler class.
    """
    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: Please set your GOOGLE_API_KEY environment variable")
        return

    # Initialize the labeler
    labeler = AudioLabeler(api_key, max_duration_seconds=10)

    # # Example 1: Analyze a single audio file
    # print("Example 1: Analyzing a single audio file")
    # print("-" * 50)
    # audio_file = "Voice_Bank/kmong/14876/audio_2.wav"

    # if audio_file and os.path.exists(audio_file):
    #     result = labeler.analyze_audio(audio_file, update_json=True)
    #     print("\nAnalysis Results:")
    #     print(json.dumps(result, indent=2))
    #     print("\nThe corresponding JSON file has been updated with age, gender, and traits.")

    # Example 2: Batch process a directory
    print("\n\nExample 2: Batch processing audio files")
    print("-" * 50)
    audio_dir = "Voice_Bank/kmong"

    if audio_dir and os.path.exists(audio_dir):
        results = labeler.batch_analyze(
            audio_dir, output_file="audio_labels.json", update_json=True
        )
        print(f"\nProcessed {len(results)} files")
        print("All corresponding JSON files have been updated.")


if __name__ == "__main__":
    main()
