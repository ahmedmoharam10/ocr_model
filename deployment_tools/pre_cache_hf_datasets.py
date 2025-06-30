# pre_cache_hf_datasets.py
import os
from datasets import load_dataset
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Temporarily override HF_DATASETS_OFFLINE to ensure download during build
os.environ['HF_DATASETS_OFFLINE'] = '0'

hf_datasets_to_load = [
    {"name": "math_qa", "trust_remote_code": True, "splits": ["train", "validation", "test"]},
    {"name": "boolq", "splits": ["train", "validation"]},  # Only has train and validation
    {"name": "squad", "config": "plain_text", "splits": ["train", "validation"]},  # Only has train and validation
    {"name": "pubmed_qa", "subset": "pqa_labeled", "splits": ["train"]},  # Only has train
    {"name": "sciq", "splits": ["train", "validation", "test"]},
    {"name": "ai2_arc", "subset": "ARC-Challenge", "splits": ["train", "validation", "test"]},
    {"name": "cais/mmlu", "subset": "college_physics", "trust_remote_code": True, "splits": ["test", "validation", "dev"]},
    {"name": "cais/mmlu", "subset": "high_school_computer_science", "trust_remote_code": True, "splits": ["test", "validation", "dev"]},
    {"name": "cais/mmlu", "subset": "college_computer_science", "trust_remote_code": True, "splits": ["test", "validation", "dev"]},
    {"name": "cais/mmlu", "subset": "electrical_engineering", "trust_remote_code": True, "splits": ["test", "validation", "dev"]},
    {"name": "openbookqa", "config": "main", "splits": ["train", "validation", "test"]},
    {"name": "lamm-mit/MechanicsMaterials", "trust_remote_code": True, "splits": ["train"]},  # Only has train
    {"name": "GainEnergy/oilandgas-engineering-dataset", "splits": ["train"]},  # Only has train
]

@retry(
    stop=stop_after_attempt(5), # Try up to 5 times
    wait=wait_exponential(multiplier=1, min=10, max=60), # **INCREASED WAIT TIMES: 10s, 20s, 40s, 60s, 60s**
    retry=retry_if_exception_type(Exception), # Retry on any exception for now
    reraise=True # Re-raise the exception if all retries fail
)
def load_and_cache_hf_dataset(name, subset=None, config=None, trust_remote_code=False, splits_to_load=None):
    """Helper function to load a single Hugging Face dataset with retry logic."""
    if splits_to_load is None:
        splits_to_load = ['train'] # Default common splits

    loaded_at_least_one_split = False
    for split in splits_to_load:
        try:
            logging.info(f"    Attempting to load split: '{split}' for {name}" + (f" (subset: {subset})" if subset else "") + (f" (config: {config})" if config else ""))
            if subset:
                load_dataset(name, subset, split=split, trust_remote_code=trust_remote_code)
            elif config:
                load_dataset(name, config, split=split, trust_remote_code=trust_remote_code)
            else:
                load_dataset(name, split=split, trust_remote_code=trust_remote_code)
            logging.info(f"    Successfully loaded split: '{split}'")
            loaded_at_least_one_split = True
        except Exception as e:
            # If it's an "Unknown split" error, just log and try the next split
            if "Unknown split" in str(e):
                logging.warning(f"    Split '{split}' not found for {name}. Trying next available split. Error: {e}")
            else:
                raise # Re-raise other types of exceptions to trigger tenacity retry

    if not loaded_at_least_one_split:
        raise Exception(f"No splits could be loaded for dataset {name}")


print("--- Starting Hugging Face Dataset Pre-caching ---")

for ds_info in hf_datasets_to_load:
    name = ds_info["name"]
    subset = ds_info.get("subset")
    config = ds_info.get("config") # Get the config from ds_info
    trust_remote_code = ds_info.get("trust_remote_code", False)
    splits = ds_info.get("splits") # Get specific splits if provided

    try:
        logging.info(f"Attempting to load and cache dataset: {name}" + (f" (subset: {subset})" if subset else "") + (f" (config: {config})" if config else ""))
        load_and_cache_hf_dataset(name, subset=subset, config=config, trust_remote_code=trust_remote_code, splits_to_load=splits)
        logging.info(f"Successfully loaded and cached: {name}" + (f" (subset: {subset})" if subset else "") + (f" (config: {config})" if config else ""))
    except Exception as e:
        logging.error(f"Failed to load and cache dataset {name}" + (f" (subset: {subset})" if subset else "") + f" after multiple retries: {e}")
        # Continue to the next dataset even if one fails

    # Add a small delay between successful dataset downloads to be polite
    time.sleep(5) # **INCREASED PAUSE BETWEEN DATASETS**

print("--- Finished Hugging Face Dataset Pre-caching ---")

# Restore HF_DATASETS_OFFLINE to its original value if needed (though Dockerfile handles final value)
os.environ['HF_DATASETS_OFFLINE'] = '1'