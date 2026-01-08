# 📘 Plagiarism Detector

**Plagiarism Detector** is a Python-based tool that compares two pieces of text to detect similarities. It provides a simple solution for identifying overlapping content and assessing potential plagiarism in documents.

---

## 🔍 Overview

The project calculates a similarity score between two input texts and highlights matching segments. It is intended for educational purposes, content validation, or preliminary plagiarism checks. Currently, it runs via a command-line interface for easy local use.

---

## ⚙️ Features

* Compare two text inputs for similarity
* Highlight overlapping text segments
* Compute a similarity score
* Fully implemented in Python with minimal dependencies

---

## 🧩 Future Enhancements

* **Web Interface:** Interactive front-end with Streamlit or Flask
* **Semantic Analysis:** NLP embeddings (e.g., BERT, SentenceTransformers) for deeper similarity detection
* **Batch Processing:** Compare multiple documents at once
* **File Support:** Accept PDFs, DOCX, and other formats
* **Reporting:** Generate downloadable reports with highlighted matches and similarity metrics

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/Aniruth1011/Plagiarism-Detector.git
cd Plagiarism-Detector
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the tool locally:

```bash
python plagiarism_detector.py
```

Enter the two texts when prompted. Output includes:

* Similarity score
* Highlighted matching segments

Example usage in Python:

```python
from Detector import detect_plagiarism

text1 = "This is the first sample text."
text2 = "This is the second sample text."

score, matches = detect_plagiarism(text1, text2)

print(f"Similarity: {score}%")
print("Matches:", matches)
```

---

## 📝 License

MIT License
