# TRAIN BASE MODELS (TRAIN SET ONLY)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

# 1. Load TRAIN dataset
print("Loading train dataset...")
train_df = pd.read_csv("data/train.csv")

X_text_data = train_df['content']
y = train_df['label']

# 2. Embeddings
print("Generating embeddings...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_text = embedder.encode(X_text_data.tolist(), show_progress_bar=True)

# 3. Text Model
print("Training text model...")
text_clf = RandomForestClassifier(n_estimators=200, random_state=42)
text_clf.fit(X_text, y)

# 4. Style Features
hindi_stopwords = [
    "है","हैं","था","थे","की","का","के","को","में",
    "पर","से","तक","और","या","लेकिन","अगर","तो",
    "ही","भी","नहीं","क्या","क्यों","कौन","कब",
    "यह","वह","इन","उन","उस","इस","कुछ","सब",
    "बहुत","अब","जब","तब","जहाँ","वहाँ"
]

def extract_style_features(text):
    words = text.split()
    sentences = text.split('.')

    word_count = len(words)
    sentence_count = len(sentences)

    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

    exclamations = text.count('!')
    questions = text.count('?')
    punctuation = len(re.findall(r'[^\w\s]', text))

    uppercase = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase / len(text) if len(text) > 0 else 0

    unique_words = len(set(words))
    ttr = unique_words / word_count if word_count > 0 else 0

    stopword_count = sum(1 for w in words if w in hindi_stopwords)
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0

    return [
        word_count,
        sentence_count,
        avg_sentence_len,
        exclamations,
        questions,
        punctuation,
        uppercase_ratio,
        ttr,
        stopword_ratio
    ]

print("Extracting style features...")
X_style = np.array([extract_style_features(t) for t in X_text_data])

# 5. Style Model
print("Training style model...")
style_clf = RandomForestClassifier(n_estimators=100, random_state=42)
style_clf.fit(X_style, y)

# 6. TF-IDF
print("Generating TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words=hindi_stopwords
)

X_tfidf = tfidf_vectorizer.fit_transform(X_text_data)

# 7. TF-IDF Model
print("Training TF-IDF model...")
tfidf_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
tfidf_clf.fit(X_tfidf, y)

# 8. Save Models
print("Saving models...")

joblib.dump(text_clf, "models/text_model.pkl")
joblib.dump(style_clf, "models/style_model.pkl")
joblib.dump(tfidf_clf, "models/tfidf_model.pkl")
joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(embedder, "models/embedding_model.pkl")

print("Base models trained!")