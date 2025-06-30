import easyocr
import torch
from PIL import Image
import numpy as np
import cv2
import re
import hashlib
import io
import os
import warnings
from typing import Dict, List
from contextlib import contextmanager
from symspellpy import SymSpell, Verbosity
from language_tool_python import LanguageTool
from ocr_utils import (
    load_dataset,
    build_domain_vocabulary,
    calculate_levenshtein_accuracy,
    calculate_domain_confidence,
    get_language_tool_instance # Import the new utility function
)

class EasyOCREngine:
    def __init__(self, vocabulary: Dict[str, List[str]], spell_checker: None):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory='/app/model_storage',
            download_enabled=False
        )
        self.vocabulary = vocabulary
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        frequency_dict_path = os.getenv("SYMSPELL_FREQ_DICT", "/app/symspell_dicts/frequency_dictionary_en_82765.txt")
        bigram_dict_path = os.getenv("SYMSPELL_BIGRAM_DICT", "/app/symspell_dicts/bigram_dictionary_en_243342.txt")
        
        if not self.sym_spell.load_dictionary(frequency_dict_path, term_index=0, count_index=1):
            print(f"Error: Frequency dictionary not loaded from {frequency_dict_path}")
        if not self.sym_spell.load_bigram_dictionary(bigram_dict_path, term_index=0, count_index=2):
            print(f"Error: Bigram dictionary not loaded from {bigram_dict_path}")
        
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
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return Image.fromarray(thresh)

    def _postprocess(self, ocr_result: List[Dict]) -> str:
        extracted_text_parts = []
        for (bbox, text, prob) in ocr_result:
            if text:
                extracted_text_parts.append(text)
        return " ".join(extracted_text_parts)

    def perform_ocr(self, image: Image.Image, ocr_quality: str = 'high') -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'PNG')
        img_hash = hashlib.md5(img_byte_arr.getvalue()).hexdigest()

        if img_hash in self.ocr_cache:
            return self.ocr_cache[img_hash]

        if ocr_quality == 'high':
            ocr_result = self.reader.readtext(np.array(image))
        else:
            ocr_result = self.reader.readtext(np.array(image))

        text = self._postprocess(ocr_result)
        self.ocr_cache[img_hash] = text
        return text

    def correct_text(self, text: str, domain: str = None) -> str:
        domain_terms = set()
        if domain and domain in self.vocabulary:
            domain_terms.update(self.vocabulary[domain])

        words_and_delimiters = re.findall(r'(\w+|[^\w\s]+|\s+)', text)
        corrected_parts = []

        for part in words_and_delimiters:
            if part.strip() == '':
                corrected_parts.append(part)
                continue

            clean_word = part.lower()

            if clean_word in domain_terms:
                corrected_parts.append(part)
            elif clean_word.isalpha() and len(clean_word) > 1:
                suggestions = self.sym_spell.lookup(
                    clean_word,
                    Verbosity.CLOSEST,
                    max_edit_distance=2,
                    include_unknown=True
                )

                if suggestions and suggestions[0].term != clean_word:
                    best_suggestion = suggestions[0].term
                    if part[0].isupper() and not part.isupper():
                        corrected_parts.append(best_suggestion.capitalize())
                    elif part.isupper():
                        corrected_parts.append(best_suggestion.upper())
                    else:
                        corrected_parts.append(best_suggestion)
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
