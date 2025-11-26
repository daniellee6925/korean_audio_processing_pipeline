# -*- coding: utf-8 -*-
from googletrans import Translator
from pathlib import Path
import json

# Recognized gender and age keywords in Korean mapped to English
gender_keywords = {"여성": "Female", "남성": "Male"}
age_keywords = {
    "10대": "10s",
    "20대": "20s",
    "30대": "30s",
    "40대": "40s",
    "50대": "50s",
    "60대": "60s",
}

# Initialize translator and translation cache
translator = Translator()
translation_cache = {}  # dictionary to store translations


def translate_term(term):
    """
    Translate a Korean term to English using the cache.
    If not in cache, call Google Translate API and store it.
    """
    if term in translation_cache:
        return translation_cache[term]
    else:
        translated = translator.translate(term, src="ko", dest="en").text
        translation_cache[term] = translated
        return translated


def translate_metadata(korean_metadata):
    """
    Translate Korean metadata and separate gender, age, and traits.
    Returns a dictionary with results.
    """
    if not korean_metadata:
        return {"gender": None, "age": None, "traits": [], "translated_list": []}

    items = [item.strip() for item in korean_metadata.split(",")]
    gender = None
    age = None
    traits = []
    translated_list = []

    for item in items:
        # Check for gender
        if item in gender_keywords:
            gender = gender_keywords[item]
            translated_list.append(gender)
        # Check for age
        elif item in age_keywords:
            age = age_keywords[item]
            translated_list.append(age)
        else:
            # Translate any other trait automatically
            translated_trait = translate_term(item)
            traits.append(translated_trait)
            translated_list.append(translated_trait)

    return {"gender": gender, "age": age, "traits": traits}


def translate_json_folder(input_folder: str, output_folder: str):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for json_file in input_path.rglob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Translate metadata if it exists
        if "metadata" in data:
            data["metadata"] = translate_metadata(data["metadata"])

        # Save to output folder with same relative path
        relative_path = json_file.relative_to(input_path)
        save_path = output_path / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Translated {json_file}")
    print(f"Finished translating {len(json_file)} files")


if __name__ == "__main__":
    translate_json_folder("data/Voice Bank/voice_casting", "data/Voice Bank/voice_casting")
