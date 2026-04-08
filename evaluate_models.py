import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

# 1. Load TEST Data ONLY
test_df = pd.read_csv("data/test.csv")
test_users = pd.read_csv("data/test_users.csv")

# 2. Load All Models
print("Loading models for evaluation...")
text_clf = joblib.load("models/text_model.pkl")
style_clf = joblib.load("models/style_model.pkl")
tfidf_clf = joblib.load("models/tfidf_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
embedder = joblib.load("models/embedding_model.pkl")

meta_model = joblib.load("models/meta_model.pkl")
meta_scaler = joblib.load("models/meta_scaler.pkl")

user_reliability_model = joblib.load("models/user_reliability_model.pkl")
social_meta_model = joblib.load("models/social_meta_model.pkl")
social_meta_scaler = joblib.load("models/social_meta_scaler.pkl")

# 3. Import feature extraction
from train_user_model import extract_user_features

# 4. Hindi Style Features
hindi_stopwords = set(["है","हैं","था","थे","की","का","के","को","में","पर","से","तक","और","या","लेकिन","अगर","तो","ही","भी","नहीं","क्या","क्यों","कौन","कब","यह","वह","इन","उन","उस","इस","कुछ","सब","बहुत","अब","जब","तब","जहाँ","वहाँ"])

def extract_style_features(text):
    words = text.split()
    sentences = text.split('.')
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
    exclamations = text.count('!')
    questions = text.count('?')
    punctuation = len(re.findall(r'[^\w\s]', text))
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    ttr = len(set(words)) / word_count if word_count > 0 else 0
    stopword_ratio = sum(1 for w in words if w in hindi_stopwords) / word_count if word_count > 0 else 0
    return [word_count, sentence_count, avg_sentence_len, exclamations, questions, punctuation, uppercase_ratio, ttr, stopword_ratio]

print("Generating base model predictions for Test News...")
X_embed_test = embedder.encode(test_df['content'].tolist(), show_progress_bar=True)
X_style_test = np.array([extract_style_features(t) for t in test_df['content']])
X_tfidf_test = tfidf_vectorizer.transform(test_df['content'])

text_probs = text_clf.predict_proba(X_embed_test)[:, 1]
style_probs = style_clf.predict_proba(X_style_test)[:, 1]
tfidf_probs = tfidf_clf.predict_proba(X_tfidf_test)[:, 1]

def print_metrics(y_true, y_pred, name):
    print(f"\n--- Metrics for {name} ---")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_true, y_pred):.4f}")

# 5. Evaluate Standard Meta Model (Unseen News)
X_meta = np.column_stack((text_probs, style_probs, tfidf_probs))
X_meta_scaled = meta_scaler.transform(X_meta)
y_pred_standard = meta_model.predict(X_meta_scaled)
print_metrics(test_df['label'], y_pred_standard, "Standard Website Model (Unseen News)")

# 6. Evaluate Social Media Meta Model (Unseen News + Unseen Users)
print("\nCalculating user scores for Test Users ONLY...")
all_test_user_scores = user_reliability_model.predict_proba(extract_user_features(test_users))[:, 1]
real_test_user_scores = all_test_user_scores[test_users['is_fake'] == 0]
fake_test_user_scores = all_test_user_scores[test_users['is_fake'] == 1]

user_scores = []
for label in test_df['label']:
    if label == 0: user_scores.append(np.random.choice(real_test_user_scores))
    else: user_scores.append(np.random.choice(fake_test_user_scores))

X_social_meta = np.column_stack((text_probs, style_probs, tfidf_probs, user_scores))
X_social_meta_scaled = social_meta_scaler.transform(X_social_meta)
y_pred_social = social_meta_model.predict(X_social_meta_scaled)
print_metrics(test_df['label'], y_pred_social, "Social Media Model (Unseen News + Unseen Users)")
