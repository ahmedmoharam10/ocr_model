import os
import fitz
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from language_tool_python import LanguageTool
from spellchecker import SpellChecker
from tesseract_ocr import TesseractEngine
from easy_ocr import EasyOCREngine
from google_ocr import GoogleVisionEngine, OCRConfig
from domain_postprocessor import DomainPostProcessor
from ocr_utils import load_dataset, build_domain_vocabulary, enhance_spellchecker, hf_load_and_extract_vocabulary, get_language_tool_instance # Import the new utility function

# --- GLOBAL INITIALIZATION (Runs once when the service starts) ---

print("Initializing OCR service: Loading domain vocabulary and language tools...")

hf_datasets_to_load = [
    {"name": "math_qa", "trust_remote_code": True},
    {"name": "boolq"},
    {"name": "squad", "config": "plain_text"},
    {"name": "pubmed_qa", "subset": "pqa_labeled"},
    {"name": "sciq"},
    {"name": "ai2_arc", "subset": "ARC-Challenge"},
    {"name": "cais/mmlu", "subset": "college_physics", "trust_remote_code": True},
    {"name": "cais/mmlu", "subset": "high_school_computer_science", "trust_remote_code": True},
    {"name": "cais/mmlu", "subset": "college_computer_science", "trust_remote_code": True},
    {"name": "cais/mmlu", "subset":"electrical_engineering", "trust_remote_code": True},
    {"name": "openbookqa", "config": "main"},
    {"name": "lamm-mit/MechanicsMaterials", "trust_remote_code": True, "subset": "default"},
    {"name": "GainEnergy/oilandgas-engineering-dataset"},
]

overall_vocabulary = {}
for ds_info in hf_datasets_to_load:
    name = ds_info["name"]
    subset = ds_info.get("subset")
    config = ds_info.get("config")
    trust_remote_code = ds_info.get("trust_remote_code", False)

    text_cols = None
    if name == "math_qa":
        text_cols = ['problem', 'question', 'answer']
    elif name == "boolq":
        text_cols = ['question', 'passage']
    elif name == "squad":
        text_cols = ['question', 'context']
    elif name == "pubmed_qa":
        text_cols = ['question', 'long_answer', 'context']
    elif name == "sciq":
        text_cols = ['question', 'support', 'distractor1', 'distractor2', 'distractor3', 'correct_answer']
    elif name == "ai2_arc":
        text_cols = ['question', 'choices']
    elif name == "openbookqa":
        text_cols = ['question_stem', 'choices', 'fact1']
    elif name == "lamm-mit/MechanicsMaterials":
        text_cols = ['text']
    elif name == "GainEnergy/oilandgas-engineering-dataset":
        text_cols = ['text']
        
    print(f"Loading Hugging Face dataset: {name}" + (f" (subset: {subset})" if subset else "") + (f" (config: {config})" if config else ""))
    
    domain_vocab = hf_load_and_extract_vocabulary(
        name,
        subset=subset,
        config=config,
        text_columns=text_cols,
        trust_remote_code=trust_remote_code
    )
    overall_vocabulary.update(domain_vocab)

post_processor = DomainPostProcessor()
common_spell_checker = SpellChecker()
enhance_spellchecker(common_spell_checker, overall_vocabulary)
print(f"SpellChecker enhanced with {len(overall_vocabulary)} domain terms.")

# Initialize LanguageTool once using the shared utility function
print("Initializing LanguageTool...")
try:
    global grammar_tool
    grammar_tool = get_language_tool_instance()
    if grammar_tool:
        print("LanguageTool initialized successfully.")
    else:
        print("LanguageTool instance is None. Grammar correction will not be available.")
except Exception as e:
    print(f"Failed to initialize LanguageTool: {e}. Grammar correction will not be available.")
    grammar_tool = None

# Initialize OCR engines once
config = OCRConfig()
easyocr_engine = EasyOCREngine(vocabulary=overall_vocabulary, spell_checker=None)
google_vision_engine = GoogleVisionEngine(config=config, vocabulary=overall_vocabulary, spell_checker=common_spell_checker)
tesseract_engine = TesseractEngine(vocabulary=overall_vocabulary, spell_checker=common_spell_checker)
print("OCR engines initialized.")

# --- END GLOBAL INITIALIZATION ---

def process_pdf_with_fallback(pdf_path: str, output_dir: str = None):
    page_results = []

    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)

        for i, page in enumerate(tqdm(doc, desc="Pages")):
            pix = page.get_pixmap(dpi=150, colorspace="rgb", alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            raw_text = None
            engine_used = "None"

            try:
                print(f"Page {i+1}: Attempting with EasyOCR...")
                raw_text = easyocr_engine.perform_ocr(img)
                engine_used = "EasyOCR"
                if raw_text and raw_text.strip():
                    print(f"Page {i+1}: EasyOCR successful.")
                else:
                    print(f"Page {i+1}: EasyOCR returned empty. Falling back.")
                    raw_text = None
            except Exception as e:
                print(f"Page {i+1}: EasyOCR failed ({e}). Falling back to Google Vision.")
                raw_text = None

            if raw_text is None or not raw_text.strip():
                try:
                    print(f"Page {i+1}: Attempting with Google Vision...")
                    raw_text = google_vision_engine.run(img)
                    engine_used = "Google Vision"
                    if raw_text and raw_text.strip():
                        print(f"Page {i+1}: Google Vision successful.")
                    else:
                        print(f"Page {i+1}: Google Vision returned empty. Falling back.")
                        raw_text = None
                except Exception as e:
                    print(f"Page {i+1}: Google Vision failed ({e}). Falling back to Tesseract.")
                    raw_text = None

            if raw_text is None or not raw_text.strip():
                try:
                    print(f"Page {i+1}: Attempting with Tesseract...")
                    raw_text = tesseract_engine.run(img)
                    engine_used = "Tesseract"
                    if raw_text and raw_text.strip():
                        print(f"Page {i+1}: Tesseract successful.")
                    else:
                        print(f"Page {i+1}: Tesseract returned empty.")
                        raw_text = ""
                except Exception as e:
                    print(f"Page {i+1}: Tesseract failed ({e}).")
                    raw_text = ""

            if raw_text:
                if engine_used == "Google Vision" or engine_used == "Tesseract":
                    corrected_text = google_vision_engine.correct_spelling(raw_text) if engine_used == "Google Vision" else tesseract_engine.correct_spelling(raw_text)
                else:
                    corrected_text = easyocr_engine.correct_text(raw_text)
                
                grammar_issues = []
                if grammar_tool: # Check if grammar_tool was successfully initialized
                    try:
                        grammar_issues = grammar_tool.check(corrected_text)
                    except Exception as e:
                        print(f"Warning: Grammar checking failed for page {i+1}: {e}")
                
                print(f"Page {i+1} | Engine Used: {engine_used} | Grammar Issues Count: {len(grammar_issues)}")
            else:
                corrected_text = ""
                grammar_issues = []
                print(f"Page {i+1} | Engine Used: {engine_used} | No text extracted.")

            page_results.append({
                "page_number": i + 1,
                "raw_text": raw_text or "",
                "corrected_text": corrected_text or "",
                "engine_used": engine_used,
                "grammar_issues_count": len(grammar_issues) if grammar_issues else 0
            })
            
        return page_results
        
    except Exception as e:
        print(f"Error in process_pdf_with_fallback: {str(e)}")
        return []
