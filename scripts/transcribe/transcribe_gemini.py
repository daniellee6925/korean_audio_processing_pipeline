import os
import asyncio
from google import genai  # Google GenAI SDK
from google.genai import types

# Constants — update as needed
MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"
PROMPT = "Generate a transcript of the speech."
MAX_CONCURRENT = 4


class GeminiTranscriber:
    def __init__(self, api_key=None):
        # If using API key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        self.model = MODEL

    async def upload_and_transcribe(self, audio_path):
        """Uploads a single audio file and asks Gemini to transcribe it."""
        # Upload file
        uploaded = await asyncio.to_thread(self.client.files.upload, file=audio_path)
        # Build contents: first the prompt, then audio part
        parts = [
            types.Part.from_text(PROMPT),
            types.Part.from_uri(uploaded.uri, uploaded.mime_type),
        ]
        contents = types.UserContent(parts)
        # Call generate_content
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=[contents],
        )
        # Assume one candidate
        text = response.candidates[0].content.text
        return text

    async def transcribe_batch(self, audio_paths):
        """Transcribe a batch of audio files concurrently."""
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def sem_task(p):
            async with sem:
                return p, await self.upload_and_transcribe(p)

        tasks = [sem_task(p) for p in audio_paths]
        result = await asyncio.gather(*tasks)
        return result


async def main(audio_dir):
    paths = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg"))
    ]
    transcriber = GeminiTranscriber(api_key=os.getenv("GOOGLE_API_KEY"))
    results = await transcriber.transcribe_batch(paths)
    for path, text in results:
        print(f"File: {path}\nTranscription:\n{text}\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.audio_dir))
