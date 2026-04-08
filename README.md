# 🇮🇳 Hindi Fake News Detection System

A comprehensive, multi-layered machine learning system designed to detect fake news in Hindi. This project uses a **Stacking Classifier** architecture that combines semantic deep learning, stylistic linguistic analysis, traditional TF-IDF keyword patterns, and **social media user reliability metrics**.

---

## 🚀 Key Features

*   **Multi-Model Ensemble**: Combines three specialized base models for text analysis.
*   **Social Media Integration**: Predicts news veracity by factoring in the credibility of the user sharing the content.
*   **Explainable AI (XAI)**: Uses LIME and SHAP to highlight *why* a piece of news was flagged as fake.
*   **Interactive Web Interface**: A Flask-based dashboard for real-time testing in both "Website" (text-only) and "Social Media" (text + user profile) modes.

---

## 📂 Project Structure

```text
├── app/                    # Web application (Flask)
│   ├── app.py              # Flask server
│   └── templates/          # HTML templates
├── data/                   # Dataset files (CSV)
│   ├── hindi_fake_news.csv # Raw fake news data
│   ├── hindi_true_news.csv # Raw true news data
│   ├── users.csv           # Real user profiles
│   ├── fusers.csv          # Fake/Bot user profiles
│   ├── train.csv / test.csv # Split news datasets
│   └── train_users.csv     # Split user datasets
├── models/                 # Saved model weights (.pkl)
├── create_dataset.py       # News data preprocessing
├── create_user_split.py    # User data preprocessing
├── train_models.py         # Trains the 3 base text models
├── train_user_model.py     # Trains the user reliability model
├── train_meta_model.py     # Trains the standard ensemble
├── train_social_meta_model.py # Trains the social-aware ensemble
├── evaluate_models.py      # Comprehensive performance metrics
├── explainability.py       # LIME & SHAP implementation
├── model_pipeline.py       # Unified inference logic
├── predict.py              # CLI testing script
└── requirements.txt        # Dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ArnavJoshi14/hindi-fake-news-detection
   cd hindi-fake-news-detection
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Unix/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Training Workflow

To build the entire system from scratch, run these scripts in order:

1.  **Data Preparation**:
    ```bash
    python create_dataset.py
    python create_user_split.py
    ```

2.  **Train Base Components**:
    ```bash
    python train_models.py      # Semantic, Style, and TF-IDF models
    python train_user_model.py  # User Reliability model
    ```

3.  **Train Ensemble (Meta) Models**:
    ```bash
    python train_meta_model.py        # Standard (Text-only)
    python train_social_meta_model.py # Social-Media aware
    ```

4.  **Evaluate**:
    ```bash
    python evaluate_models.py
    ```

---

## 🔮 Usage

### 1. Web Dashboard (Recommended)
Launch the interactive interface to test articles:
```bash
python app/app.py
```
Open `http://localhost:5000` in your browser.

### 2. CLI Prediction
Test individual articles via terminal:
```bash
python predict.py
```

### 3. Explainability
To generate visual explanations (LIME/SHAP plots) for a specific text, you can call functions within `explainability.py`.

---

## 🧠 Model Architecture

### Base Models:
1.  **Semantic (SBERT)**: Uses `multilingual-MiniLM-L12-v2` to understand the deep contextual meaning of Hindi text.
2.  **Stylistic**: Analyzes 9 linguistic features (stopword ratio, punctuation density, sentence complexity).
3.  **TF-IDF**: Captures N-gram patterns and specific keyword associations.

### Meta Models:
*   **Standard Meta**: A Logistic Regression model that weighs the probabilities from the three base models.
*   **Social Meta**: Adds a 4th input—the **User Reliability Score**—to the ensemble, providing a more holistic "Social Media" veracity check.

---

## 🧰 Dependencies

*   `scikit-learn`: Core ML algorithms
*   `sentence-transformers`: Multi-lingual embeddings
*   `pandas` & `numpy`: Data processing
*   `flask`: Web framework
*   `lime` & `shap`: Explainability
*   `matplotlib`: Visualization
*   `joblib`: Model serialization
