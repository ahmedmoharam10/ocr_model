# pre_cache_languagetool.py
import language_tool_python
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pre_cache_languagetool():
    """
    Downloads and pre-caches the LanguageTool data into the specified directory.
    This script is intended to be run during the Docker build process.
    """
    # Ensure the directory exists and set it as the default cache location
    # This must match the DEFAULT_LANGUAGE_TOOL_DIR set in ocr_utils.py
    # and the ENV variable in the Dockerfile for consistency.
    cache_dir = os.getenv("LANGUAGE_TOOL_PYTHON_DIR", "/app/languagetool_cache")
    os.makedirs(cache_dir, exist_ok=True)
    language_tool_python.download_lt.DEFAULT_LANGUAGE_TOOL_DIR = cache_dir

    logging.info(f"Starting LanguageTool pre-caching to: {cache_dir}")
    try:
        # Initialize LanguageTool to ensure all necessary language data is cached
        logging.info("Initializing LanguageTool to cache language data...")
        # Note: 'en-US' is a common default; if you use other languages, consider pre-caching them too.
        tool_instance = language_tool_python.LanguageTool('en-US')
        
        # Perform a dummy check to ensure full load and trigger all downloads
        _ = tool_instance.check("hello world") 
        
        logging.info("LanguageTool pre-cached successfully.")
    except Exception as e:
        logging.error(f"Failed to pre-cache LanguageTool: {e}")
        # Re-raise to fail the Docker build if pre-caching fails
        raise

if __name__ == "__main__":
    pre_cache_languagetool()
