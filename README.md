# Robust Phishing URL Detection

A machine learning framework for detecting malicious and phishing URLs with robust cross-domain evaluation.

---

## 📌 Overview

Traditional phishing URL classifiers often suffer from synthetic high performance due to domain leakage between training and testing splits. This repository implements **Cross-Domain Phishing URL Detection**, ensuring that domains present in the training set never appear in the testing set. This simulates real-world zero-day phishing scenarios where unseen domain names are encountered.

### Key Highlights
- **Cross-Domain Split**: Evaluates the model on unseen domain names to measure true generalization capability.
- **Character N-Gram TF-IDF**: Uses sublinear term frequency character n-grams (ranges 2–5) to capture deceptive subword patterns, typosquatting, and malicious tokens.
- **Class-Balanced Logistic Regression**: Optimized for high recall and precision on imbalanced URL distributions.
- **Evaluation & Diagnostics**: Generates classification reports, confusion matrix heatmaps, and probability scores for single URL predictions.

---

## 📂 Project Structure

```text
Robust-Phishing-URL-Detection/
│
├── data/
│   └── .gitkeep                 # Place phishing_site_urls.csv here
├── main.py                      # Training, cross-domain evaluation, and inference
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore configuration
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/padalakarthikeya/Robust-Phishing-URL-Detection.git
   cd Robust-Phishing-URL-Detection
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Dataset & Usage

### 1. Dataset Setup
Download the phishing URL dataset (e.g. `phishing_site_urls.csv`) and place it in the `data/` directory:
```text
data/phishing_site_urls.csv
```

### 2. Run Training and Evaluation
Execute `main.py` to preprocess data, perform domain-disjoint splitting, extract character n-gram TF-IDF features, train the model, and evaluate results:
```bash
python main.py
```

### 3. Making Predictions
You can use the built-in `predict_url()` function in `main.py`:
```python
predict_url("https://www.google.com")
predict_url("http://paypal-login-security-verify.xyz/login")
```

---

## 📊 Evaluation Metrics

The pipeline outputs:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix Visualization**
- **Zero-division Safe Classification Report**
