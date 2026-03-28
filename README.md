# Hindi Fake News Detection

A robust, multi-layered machine learning system designed to detect fake news in Hindi. This project employs an ensemble (meta-model) approach, combining semantic embeddings, stylistic features, and traditional TF-IDF analysis to achieve high accuracy.

## 🚀 Overview

The detector uses a **Stacking Classifier** architecture. It consists of three base models whose outputs are fed into a final meta-model (Logistic Regression) to make the ultimate decision.

### Base Models:
1.  **Semantic Model**: Uses `SentenceTransformer` (multilingual-MiniLM) to capture the deep meaning of the text.
2.  **Stylistic Model**: Analyzes linguistic patterns such as word count, sentence length, punctuation usage, and stopword ratios.
3.  **TF-IDF Model**: Traditional N-gram analysis to capture specific keyword patterns common in fake or real news.

## 📂 Project Structure

```text
├── data/                   # Dataset files (CSV)
│   ├── hindi_fake_news.csv # Raw fake news data
│   ├── hindi_true_news.csv # Raw true news data
│   ├── train.csv           # Processed training split
│   └── test.csv            # Processed testing split
├── models/                 # Saved model weights (.pkl)
├── create_dataset.py       # Data preprocessing and splitting
├── train_models.py         # Training the three base models
├── train_meta_model.py     # Training the final ensemble model
├── model_pipeline.py       # Unified pipeline for inference
├── predict.py              # CLI script for testing individual articles
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ArnavJoshi14/hindi-fake-news-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Training Workflow

To train the system from scratch, follow these steps in order:

1.  **Prepare the Dataset**:
    ```bash
    python create_dataset.py
    ```
    This combines the raw CSVs into `train.csv` and `test.csv`.

2.  **Train Base Models**:
    ```bash
    python train_models.py
    ```
    This trains the Semantic, Stylistic, and TF-IDF models and saves them to the `models/` directory.

3.  **Train Meta Model**:
    ```bash
    python train_meta_model.py
    ```
    This generates predictions from base models on the training set and trains the final Logistic Regression meta-model. It also outputs evaluation metrics (Accuracy, F1, ROC-AUC).

## 🔮 Usage (Inference)

You can test the model on any Hindi news article using the `predict.py` script:

```bash
python predict.py
```
When prompted, paste the Hindi text of the article. The system will provide:
*   Final Prediction (REAL or FAKE)
*   Confidence Score
*   Detailed breakdown of scores from each individual base model.

## 🧰 Dependencies

*   `pandas` & `numpy`: Data manipulation
*   `scikit-learn`: Machine learning algorithms and preprocessing
*   `sentence-transformers`: Multilingual text embeddings
*   `torch`: Backend for transformer models
*   `joblib`: Model serialization
*   `flask`: (Optional) For web deployment
