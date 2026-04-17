# 🇮🇳 Hindi Fake News Detection with User Credibility

A comprehensive, multi-layered machine learning system designed to detect fake news in **Hindi**. This project leverages a **stacking classifier architecture** that integrates semantic deep learning, linguistic style analysis, TF-IDF keyword extraction, and **social media user credibility scoring** for context-aware predictions.

> 🔗 **GitHub Repository:** [github.com/ketangarg15/Hindi_Fake_News_Detection_with_user_credibility](https://github.com/ketangarg15/Hindi_Fake_News_Detection_with_user_credibility)

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🧠 Multi-Model Ensemble | Stacking classifier combining semantic, stylistic, and keyword models |
| 👤 Social Media Awareness | User credibility scoring to detect bot/fake profiles |
| 🔍 Explainable AI (XAI) | LIME & SHAP for transparent, interpretable predictions |
| 🌐 Interactive Web Dashboard | Flask app with Website Mode and Social Media Mode |
| 🇮🇳 Hindi Language Support | Multilingual SBERT embeddings optimized for Hindi text |

---

## 🧠 Model Architecture

```
Input: Hindi News Article  [+ Optional: User Profile]
              │
              ▼
   ┌──────────────────────┐
   │  Multilingual SBERT  │  (multilingual-MiniLM-L12-v2)
   └──────────────────────┘
              │
   ┌──────────┼──────────────┐
   ▼          ▼              ▼
Semantic   Stylistic      TF-IDF
 Model      Model          Model
(SBERT)  (Linguistic    (N-gram +
          Features)      Keywords)
   │          │              │
   └──────────┼──────────────┘
              │
    ┌─────────┴──────────────────────┐
    │                                │
    ▼                                ▼
Standard Meta Model          Social Meta Model
(Logistic Regression)        (Logistic Regression)
Text signals only            Text + User Credibility Score
    │                                │
    └─────────────┬──────────────────┘
                  ▼
         Final Prediction
          (FAKE / REAL)
```

---

## 🔹 Base Models

### Semantic Model (SBERT)
- Uses `multilingual-MiniLM-L12-v2` from Sentence Transformers
- Captures deep contextual and semantic meaning of Hindi text
- Language-agnostic embeddings handle Hindi script natively

### Stylistic Model
Extracts linguistic surface features including:
- Stopword ratio and function word frequency
- Punctuation density and capitalization patterns
- Sentence complexity and average word length
- Vocabulary diversity metrics

### TF-IDF Model
- Captures keyword patterns and N-gram frequency distributions
- Identifies manipulation signals through term weighting
- Complements semantic model with lexical-level features

## 🔹 Meta Models

### Standard Meta Model
- Logistic Regression combining outputs of the three text-based models
- Used for **Website Mode** (text-only analysis)

### Social Meta Model
- Extends the Standard Meta Model with an additional **User Reliability Score**
- Used for **Social Media Mode** (text + user profile analysis)
- Enables context-aware detection accounting for source credibility

---

## 📂 Project Structure

```
hindi-fake-news-detection/
│
├── app/
│   ├── app.py                       # Flask application server
│   └── templates/                   # HTML templates
│
├── data/
│   ├── hindi_fake_news.csv          # Fake news dataset
│   ├── hindi_true_news.csv          # True news dataset
│   ├── users.csv                    # Genuine user profiles
│   ├── fusers.csv                   # Fake / bot profiles
│   ├── train.csv / test.csv         # News train-test splits
│   └── train_users.csv              # User data splits
│
├── models/                          # Saved trained models (.pkl)
│
├── create_dataset.py                # News data preprocessing
├── create_user_split.py             # User data preprocessing
├── train_models.py                  # Train base text models
├── train_user_model.py              # Train user credibility model
├── train_meta_model.py              # Train text-only ensemble
├── train_social_meta_model.py       # Train social-aware ensemble
├── evaluate_models.py               # Performance evaluation
├── explainability.py                # LIME & SHAP implementation
├── model_pipeline.py                # Unified inference pipeline
├── predict.py                       # CLI prediction script
└── requirements.txt                 # Python dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ketangarg15/Hindi_Fake_News_Detection_with_user_credibility.git
cd Hindi_Fake_News_Detection_with_user_credibility
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Place the required files in the `data/` directory:

| File | Description |
|---|---|
| `hindi_fake_news.csv` | Labeled fake Hindi news articles |
| `hindi_true_news.csv` | Labeled real Hindi news articles |
| `users.csv` | Genuine social media user profiles |
| `fusers.csv` | Fake / bot user profiles |
| `train.csv` / `test.csv` | Pre-split news data |
| `train_users.csv` | Pre-split user data |

> **Note:** Each news file should contain at minimum a `text` column with the article body and a `label` column (`0` = Real, `1` = Fake).

---

## 📈 Training Workflow

Run the following scripts **in order** to build the full system from scratch.

### Step 1 — Data Preparation

```bash
python create_dataset.py       # Preprocess and merge news datasets
python create_user_split.py    # Preprocess and split user profiles
```

### Step 2 — Train Base Models

```bash
python train_models.py         # Trains Semantic, Style, and TF-IDF models
python train_user_model.py     # Trains User Credibility model
```

### Step 3 — Train Meta Models

```bash
python train_meta_model.py          # Text-only ensemble (Standard)
python train_social_meta_model.py   # Social-aware ensemble (with User Score)
```

### Step 4 — Evaluate

```bash
python evaluate_models.py
```

### Evaluation Output Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted fakes, how many were actually fake |
| **Recall** | Of actual fakes, how many were correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall |

---

## 🔮 Usage

### 1. Web Interface (Recommended)

Launch the Flask application:

```bash
python app/app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

The dashboard supports two modes:

- **Website Mode** — Paste a Hindi news article for text-only analysis
- **Social Media Mode** — Provide both article text and a user profile for credibility-aware analysis

### 2. Command-Line Prediction

```bash
python predict.py
```

Follow the prompts to input a Hindi article and (optionally) a user profile for a quick CLI prediction.

### 3. Explainability

To generate LIME or SHAP explanations for any prediction:

```bash
python explainability.py
```

This outputs feature-level importance scores showing which words or patterns most influenced the prediction — making the model's decision interpretable.

---

## 🧰 Dependencies

| Library | Purpose |
|---|---|
| `scikit-learn` | ML algorithms (Random Forest, Logistic Regression) |
| `sentence-transformers` | Multilingual SBERT embeddings |
| `pandas`, `numpy` | Data processing and manipulation |
| `flask` | Web application framework |
| `lime`, `shap` | Explainability tools |
| `matplotlib` | Visualization |
| `joblib` | Model serialization |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🎯 Key Advantages

- **Multilingual-ready** — SBERT embeddings handle Hindi script without manual transliteration
- **Source-aware** — User credibility scoring adds a social trust layer beyond text analysis
- **Explainable** — LIME/SHAP integration makes predictions interpretable and auditable
- **Dual-mode** — Supports both pure-text and social media contexts in one system
- **Modular** — Each model component can be retrained or swapped independently

---

## ⚠️ Limitations

- Performance depends on the quality and diversity of the Hindi news dataset
- User credibility model requires social media profile data — not always available
- May underperform on highly informal Hindi or code-mixed Hindi-English text
- LIME/SHAP explanations add inference latency for real-time use cases

---

## 🚀 Future Improvements

- [ ] Fine-tune on a larger Hindi-specific corpus (e.g., IndicNLP, AI4Bharat)
- [ ] Support code-mixed Hindi-English text (Hinglish)
- [ ] Integrate live news APIs for real-time KB updates
- [ ] Add multi-class labels (Satire / Misleading / False / True)
- [ ] Dockerize for easy deployment
- [ ] Build a REST API for programmatic access
- [ ] Extend to other Indian languages (Tamil, Bengali, Marathi)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Ketan Garg**
- GitHub: [@ketangarg15](https://github.com/ketangarg15)
- Repository: [Hindi_Fake_News_Detection_with_user_credibility](https://github.com/ketangarg15/Hindi_Fake_News_Detection_with_user_credibility)

---

## 🙏 Acknowledgements

- [Hugging Face Sentence Transformers](https://www.sbert.net/) for multilingual SBERT embeddings
- [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap) for explainability frameworks
- [Scikit-learn](https://scikit-learn.org/) for ML utilities
- AI4Bharat and IndicNLP community for Hindi NLP resources
