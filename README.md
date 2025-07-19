
# 🤖 OCR Model – Robust Multi-Engine, Domain-Aware OCR Pipeline

A modular OCR pipeline designed for reliable text extraction from PDFs and images using multiple fallback engines, domain-specific enhancements, and ready-to-deploy cloud infrastructure.

---

## 🌟 Features

### 🔁 Three-Tier OCR Engine
- **Primary**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Fallbacks**:
  - Google Vision API
  - Tesseract OCR
- Automatically detects engine failures or low-confidence output and switches engines to ensure best results.

### 📄 Unified Text Extraction
- Accepts `.jpg`, `.png`, `.pdf`, `.tiff` inputs.
- Outputs:
  - Extracted structured **text**
  - **Engine used**
  - Detected **grammar/spelling issues** (optional)

### 🎓 Domain-Specific Academic Enhancements
- Recognizes educational terms using local and Hugging Face datasets.
- Detects:
  - **Academic keywords**
  - **Math/science formulas** (e.g., `E = mc^2`, `lim x→0`)
- Ideal for digitizing **exams**, **lecture notes**, **scientific papers**, etc.

---

## 📁 Folder Structure

```
ocr_model/
├── main.py
├── ocr_engines/
│   ├── easyocr_engine.py
│   ├── tesseract_engine.py
│   └── vision_engine.py
├── domain_enhancement.py
├── grammar_check.py
├── pdf_utils.py
├── config.py
├── deployment_tools/
│   ├── Dockerfile
│   └── requirements.txt
└── assets/
```

---

## ⚙️ Installation

### 🧰 Local Setup
```bash
git clone https://github.com/ahmedmoharam10/ocr_model.git
cd ocr_model
pip install -r deployment_tools/requirements.txt
```

### 🔧 External Tools
- **Tesseract**: Install and add to PATH → [Install Guide](https://github.com/tesseract-ocr/tesseract)
- **Google Vision API**:
  - Enable Vision API on Google Cloud
  - Download credentials JSON
  - Set env variable:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/creds.json"
    ```

---

## 🚀 Usage

```bash
python main.py --input_file path/to/image_or_pdf
```


---

## ☁️ Cloud Deployment

### 🐳 Docker (Ready for GCP, AWS, Railway, Render)

```bash
# Build the Docker image
docker build -t ocr-model -f deployment_tools/Dockerfile .

# Run the container locally
docker run -p 5000:5000 ocr-model
```

> You can easily deploy this container to Google Cloud Run, AWS ECS/Fargate, or any cloud platform that supports Docker.

---

## 🧠 Use Cases

- 📚 Academic handouts, scanned lecture notes
- 🧪 Scientific papers with math formulas
- 🧾 Medical prescriptions with handwritten text
- 🏫 Exams, assignments, and reports

---

## 📜 License

MIT License © [Ahmed Moharam](https://github.com/ahmedmoharam10)

---

