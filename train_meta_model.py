# TRAIN META MODEL (FINAL WITH SCALING + UPDATED FEATURES)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import re

# 1. Load models
print("Loading models...")

text_clf = joblib.load("models/text_model.pkl")
style_clf = joblib.load("models/style_model.pkl")
tfidf_clf = joblib.load("models/tfidf_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
embedder = joblib.load("models/embedding_model.pkl")

# 2. Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 3. Hindi Stopwords
hindi_stopwords = set([
    "है","हैं","था","थे","की","का","के","को","में",
    "पर","से","तक","और","या","लेकिन","अगर","तो",
    "ही","भी","नहीं","क्या","क्यों","कौन","कब",
    "यह","वह","इन","उन","उस","इस","कुछ","सब",
    "बहुत","अब","जब","तब","जहाँ","वहाँ"
])

# 4. Style Feature Function 
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

# 5. Generate TRAIN meta features
print("Generating TRAIN meta features...")

X_embed_train = embedder.encode(train_df['content'].tolist(), show_progress_bar=True)
X_style_train = np.array([extract_style_features(t) for t in train_df['content']])
X_tfidf_train = tfidf_vectorizer.transform(train_df['content'])

text_pred_train = text_clf.predict_proba(X_embed_train)[:, 1]
style_pred_train = style_clf.predict_proba(X_style_train)[:, 1]
tfidf_pred_train = tfidf_clf.predict_proba(X_tfidf_train)[:, 1]

X_meta_train = np.column_stack((text_pred_train, style_pred_train, tfidf_pred_train))
y_train = train_df['label']

# 6. Generate TEST meta features
print("Generating TEST meta features...")

X_embed_test = embedder.encode(test_df['content'].tolist(), show_progress_bar=True)
X_style_test = np.array([extract_style_features(t) for t in test_df['content']])
X_tfidf_test = tfidf_vectorizer.transform(test_df['content'])

text_pred_test = text_clf.predict_proba(X_embed_test)[:, 1]
style_pred_test = style_clf.predict_proba(X_style_test)[:, 1]
tfidf_pred_test = tfidf_clf.predict_proba(X_tfidf_test)[:, 1]

X_meta_test = np.column_stack((text_pred_test, style_pred_test, tfidf_pred_test))
y_test = test_df['label']

# 7. SCALING 
scaler = StandardScaler()
X_meta_train = scaler.fit_transform(X_meta_train)
X_meta_test = scaler.transform(X_meta_test)

# 8. Train meta model
print("Training meta model...")
meta_model = LogisticRegression(C=0.3)
meta_model.fit(X_meta_train, y_train)

# 9. Evaluation
print("\nEvaluating...")

y_pred = meta_model.predict(X_meta_test)
y_proba = meta_model.predict_proba(X_meta_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Save models
joblib.dump(meta_model, "models/meta_model.pkl")
joblib.dump(scaler, "models/meta_scaler.pkl")

print("\n Meta model trained!")