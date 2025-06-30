import re
import json
from typing import Dict, List
from rapidfuzz import fuzz # For Levenshtein distance calculation

class DomainPostProcessor:
    def __init__(self, dataset_path: str = "/app/datasets/dataset.json"):
        self.domain_patterns = self._load_domain_patterns(dataset_path)
    
    def _load_domain_patterns(self, dataset_path: str) -> Dict[str, List[str]]:
        """
        Extract common patterns (equations, formulas) for each domain from the dataset.
        """
        from collections import defaultdict
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at '{dataset_path}'. Cannot load domain patterns.")
            return defaultdict(list)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{dataset_path}'. Cannot load domain patterns.")
            return defaultdict(list)
            
        patterns = defaultdict(list)
        for entry in dataset:
            label = entry.get('label')
            text = entry.get('text', '')
            if label and text:
                # Regex to extract chemical formulas (e.g., H2O, CO2, NaCl)
                # or simple equations/expressions that look like formulas (e.g., A+B=C)
                # This regex captures sequences of CapitalLetter-optional_lowercase-optional_digits
                # OR sequences of CapitalLetter-optional_lowercase-optional_digits followed by symbols like +, -, →, =
                formula_matches = re.findall(r'([A-Z][a-z]?\d*)+|([A-Z][a-z]?\d*[\+\-→=].+)', text)
                
                for match_tuple in formula_matches:
                    # `re.findall` with multiple groups returns a tuple of strings,
                    # where only one group in the tuple will have a match and others are empty.
                    clean_match = next((m for m in match_tuple if m), None)
                    
                    if clean_match:
                        # Clean up any leading/trailing whitespace or punctuation that might have been included
                        clean_match = clean_match.strip()
                        
                        # Further refine to ensure it's a valid pattern and not just a single word
                        # Add more filtering if needed, e.g., length check, specific characters
                        if len(clean_match) > 1 and clean_match not in patterns[label]:
                            patterns[label].append(clean_match)
        return patterns
    
    def correct_domain_specific(self, text: str, domain: str) -> str:
        """
        Correct domain-specific patterns in text based on pre-loaded patterns.
        Uses Levenshtein distance for fuzzy matching.
        """
        if domain not in self.domain_patterns:
            return text
            
        # Split text into words/tokens while preserving delimiters (like spaces, punctuation)
        # This helps in replacing the exact erroneous part while maintaining text structure.
        words_and_delimiters = re.findall(r'(\w+)([^\w\s]*|\s+)', text)
        
        corrected_parts = []
        for word, delimiter in words_and_delimiters:
            replaced = False
            for pattern in self.domain_patterns[domain]:
                # Using token_set_ratio for better matching of word sequences or permutations
                # fuzz.ratio for exact string similarity
                if fuzz.ratio(word, pattern) > 85:  # 85% similarity threshold
                    corrected_parts.append(pattern + delimiter) # Replace with known correct pattern
                    replaced = True
                    break # Move to the next word once a replacement is made
            if not replaced:
                corrected_parts.append(word + delimiter) # Keep original if no match found
                
        return ''.join(corrected_parts).strip() # Join back and strip any extra whitespace