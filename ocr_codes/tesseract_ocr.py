import pytesseract
from PIL import Image
import numpy as np
import cv2
from spellchecker import SpellChecker
from language_tool_python import LanguageTool
from contextlib import contextmanager
import re
import hashlib
import io
import os
from typing import Dict, List
from ocr_utils import (
    load_dataset,
    build_domain_vocabulary,
    enhance_spellchecker,
    calculate_levenshtein_accuracy,
    calculate_domain_confidence,
    get_language_tool_instance # Import the new utility function
)

class TesseractEngine:
    def __init__(self, vocabulary: Dict[str, List[str]], spell_checker: SpellChecker):
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
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
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)

    def _get_image_hash(self, image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        if image.mode != 'L':
            image = image.convert('L')
        image.save(img_byte_arr, format='PNG') 
        return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

    def run(self, image: Image.Image) -> str:
        processed_image = self._preprocess(image)
        image_hash = self._get_image_hash(processed_image)

        if image_hash in self.ocr_cache:
            return self.ocr_cache[image_hash]

        try:
            raw_text = pytesseract.image_to_string(processed_image)
            self.ocr_cache[image_hash] = raw_text
            return raw_text
        except Exception as e:
            print(f"Error during Tesseract processing: {e}")
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
