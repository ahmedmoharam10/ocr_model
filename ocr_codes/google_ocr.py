import os
import io
from typing import Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np
from spellchecker import SpellChecker
from google.cloud import vision
from language_tool_python import LanguageTool
from contextlib import contextmanager
import re
from collections import defaultdict
from ocr_utils import (
    load_dataset,
    build_domain_vocabulary,
    enhance_spellchecker,
    calculate_levenshtein_accuracy,
    calculate_domain_confidence,
    get_language_tool_instance # Import the new utility function
)
from typing import Dict, List
import cv2

@dataclass
class OCRResult:
    raw_text: str
    corrected_text: str
    accuracy: float
    grammar_issues: int
    engine_name: str
    error: Optional[str] = None

class OCRConfig:
    def __init__(self):
        self.GOOGLE_CREDENTIALS_PATH = "/app/credentials/green-books-462600-dcadf7306d66.json"
        self.USE_GOOGLE_VISION = True

class GoogleVisionEngine:
    def __init__(self, config: OCRConfig, vocabulary: Dict[str, List[str]], spell_checker: SpellChecker):
        self.name = "GoogleVision"
        self.config = config
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.GOOGLE_CREDENTIALS_PATH
        self.client = vision.ImageAnnotatorClient()
        self.vocabulary = vocabulary
        self.spell = spell_checker
        # Use the shared utility function to initialize LanguageTool
        self.language_tool = get_language_tool_instance()
        self.ocr_cache = {}

    @contextmanager
    def language_tool_context(self):
        try:
            yield self.language_tool
        finally:
            pass

    def _preprocess(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        if img_np.shape[2] == 3 and img_np.shape[2] != image.mode.count('RGB'):
            img = Image.fromarray(img_np)
            return img.convert('RGB')
        return Image.fromarray(img_np)

    def _prepare_image_bytes(self, image: Image.Image) -> bytes:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def run(self, image: Image.Image) -> str:
        processed_image = self._preprocess(image)
        image_bytes = self._prepare_image_bytes(processed_image)

        try:
            image_vision = vision.Image(content=image_bytes)
            response = self.client.document_text_detection(image=image_vision)
            raw_text = response.full_text_annotation.text if response.full_text_annotation else ""
            return raw_text
        except Exception as e:
            print(f"Error during Google Vision API call: {e}")
            return ""

    def correct_spelling(self, text: str) -> str:
        domain_terms = set()
        for domain_list in self.vocabulary.values():
            domain_terms.update(domain_list)

        words = re.split(r'(\s+)', text)
        corrected_parts = []
        for i, part in enumerate(words):
            if part.strip() == '':
                corrected_parts.append(part)
                continue

            clean_word = re.sub(r'^\W+|\W+$', '', part).lower()

            if clean_word in domain_terms:
                corrected_parts.append(part)
            elif clean_word.isalpha() and len(clean_word) > 2:
                correction = self.spell.correction(clean_word)
                if correction is not None and correction != clean_word:
                    if part[0].isupper():
                        corrected_parts.append(correction.capitalize())
                    elif part.isupper() and len(part) > 1:
                        corrected_parts.append(correction.upper())
                    else:
                        corrected_parts.append(correction)
                else:
                    corrected_parts.append(part)
            else:
                corrected_parts.append(part)

        basic_corrected = "".join(corrected_parts)
        with self.language_tool_context() as lt:
            return lt.correct(basic_corrected)

    def check_grammar(self, text: str) -> int:
        with self.language_tool_context() as lt:
            return len(lt.check(text))

    def calculate_accuracy(self, pred: str, truth: str) -> float:
        return calculate_levenshtein_accuracy(pred, truth)
