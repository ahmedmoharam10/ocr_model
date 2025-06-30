FROM python:3.10-slim-bookworm

WORKDIR /app

RUN useradd -m appuser -u 1000

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    openjdk-17-jre-headless \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    poppler-utils \
    ghostscript \
    libtesseract-dev \
    libleptonica-dev \
    wget \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ \
    && wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/

# Copy requirements.txt from the root of the build context
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && python3 -c "import nltk; nltk.download('punkt')"

ENV HF_DATASETS_OFFLINE=0
ENV LANGUAGE_TOOL_PYTHON_DIR=/app/languagetool_cache

# Create the LanguageTool cache directory and set ownership as root
RUN mkdir -p ${LANGUAGE_TOOL_PYTHON_DIR} && chown appuser:appuser ${LANGUAGE_TOOL_PYTHON_DIR}

USER appuser

# Copy and run the LanguageTool pre-caching script as appuser (these are at the root of ai_project)
COPY --chown=appuser:appuser pre_cache_languagetool.py /tmp/
RUN python3 /tmp/pre_cache_languagetool.py \
    && rm /tmp/pre_cache_languagetool.py

# Copy and run the Hugging Face datasets pre-caching script as appuser (these are at the root of ai_project)
COPY --chown=appuser:appuser pre_cache_hf_datasets.py /tmp/
RUN python3 /tmp/pre_cache_hf_datasets.py \
    && rm /tmp/pre_cache_hf_datasets.py

USER root

# IMPORTANT CORRECTION: Adjust COPY commands to use the 'project/' subdirectory
# Copy main application Python files from the 'project/' subdirectory
COPY --chown=1000:1000 ./project/*.py ./
# Copy other necessary directories from the 'project/' subdirectory
COPY --chown=1000:1000 ./project/datasets/ /app/datasets/
COPY --chown=1000:1000 ./project/credentials/ /app/credentials/
COPY --chown=1000:1000 ./project/symspell_dicts/ /app/symspell_dicts/
COPY --chown=1000:1000 ./project/models/ /app/models/
COPY --chown=1000:1000 ./project/model_storage/ /app/model_storage/

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/green-books-462600-dcadf7306d66.json
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV SYMSPELL_FREQ_DICT=/app/symspell_dicts/frequency_dictionary_en_82765.txt
ENV SYMSPELL_BIGRAM_DICT=/app/symspell_dicts/bigram_dictionary_en_243342.txt
ENV DATASET_PATH=/app/datasets/dataset.json
# Ensure LANGUAGE_TOOL_PATH is consistent with LANGUAGE_TOOL_PYTHON_DIR for runtime
ENV LANGUAGE_TOOL_PATH=/app/languagetool_cache/LanguageTool
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OCR_QUALITY=high
ENV HF_DATASETS_OFFLINE=1
ENV TESSERACT_THREADS=1
ENV EASYOCR_MODULE_PATH=/app/model_storage

RUN ln -s /usr/bin/tesseract /usr/local/bin/tesseract \
    && chmod -R a+r /app \
    && chmod a+x /app/*.py

RUN chown -R appuser:appuser /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

USER appuser

EXPOSE 8000
CMD uvicorn Orchestration:app --host 0.0.0.0 --port $PORT --workers 1