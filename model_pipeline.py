import joblib
import numpy as np
import pandas as pd
import re

class FakeNewsPipeline:

    def __init__(self, model_dir="models/"):
        print("Loading pipeline models...")

        # Base Models
        self.text_clf = joblib.load(model_dir + "text_model.pkl")
        self.style_clf = joblib.load(model_dir + "style_model.pkl")
        self.tfidf_clf = joblib.load(model_dir + "tfidf_model.pkl")
        self.tfidf_vectorizer = joblib.load(model_dir + "tfidf_vectorizer.pkl")
        self.embedder = joblib.load(model_dir + "embedding_model.pkl")

        # Standard Meta Model
        self.meta_model = joblib.load(model_dir + "meta_model.pkl")
        self.meta_scaler = joblib.load(model_dir + "meta_scaler.pkl")

        # Social Media Specific Models
        self.user_reliability_model = joblib.load(model_dir + "user_reliability_model.pkl")
        self.social_meta_model = joblib.load(model_dir + "social_meta_model.pkl")
        self.social_meta_scaler = joblib.load(model_dir + "social_meta_scaler.pkl")

        self.hindi_stopwords = [
            "है","हैं","था","थे","की","का","के","को","में",
            "पर","से","तक","और","या","लेकिन","अगर","तो",
            "ही","भी","नहीं","क्या","क्यों","कौन","कब",
            "यह","वह","इन","उन","उस","इस","कुछ","सब",
            "बहुत","अब","जब","तब","जहाँ","वहाँ"
        ]

        print(" Pipeline ready!")

    def extract_style_features(self, text):
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

        stopword_count = sum(1 for w in words if w in self.hindi_stopwords)
        stopword_ratio = stopword_count / word_count if word_count > 0 else 0

        return [[
            word_count,
            sentence_count,
            avg_sentence_len,
            exclamations,
            questions,
            punctuation,
            uppercase_ratio,
            ttr,
            stopword_ratio
        ]]

    def extract_user_features(self, user_data):
        """
        user_data: dict containing keys like 'name', 'screen_name', 'statuses_count', etc.
        """
        features = {
            'statuses_count': user_data.get('statuses_count', 0),
            'followers_count': user_data.get('followers_count', 0),
            'friends_count': user_data.get('friends_count', 0),
            'favourites_count': user_data.get('favourites_count', 0),
            'listed_count': user_data.get('listed_count', 0),
            'verified': 1 if user_data.get('verified') in ['TRUE', True] else 0,
            'default_profile': 1 if user_data.get('default_profile') in ['TRUE', True] else 0,
            'name_len': len(str(user_data.get('name', ''))),
            'name_digits': sum(c.isdigit() for c in str(user_data.get('name', ''))),
            'screen_name_len': len(str(user_data.get('screen_name', ''))),
            'screen_name_digits': sum(c.isdigit() for c in str(user_data.get('screen_name', ''))),
            'has_location': 1 if str(user_data.get('location', '')).strip() != "" and str(user_data.get('location', '')).lower() != 'null' else 0
        }
        return pd.DataFrame([features])

    def predict(self, text, mode="website", user_data=None):
        """
        mode: "website" or "social_media"
        user_data: dict of user metadata (required if mode="social_media")
        """
        text_lower = text.lower()

        # Get base model scores
        embedding = self.embedder.encode([text_lower])
        style_features = self.extract_style_features(text_lower)
        tfidf_features = self.tfidf_vectorizer.transform([text_lower])

        text_prob = self.text_clf.predict_proba(embedding)[0][1]
        style_prob = self.style_clf.predict_proba(style_features)[0][1]
        tfidf_prob = self.tfidf_clf.predict_proba(tfidf_features)[0][1]

        if mode == "social_media" and user_data:
            # Get user reliability score
            user_features = self.extract_user_features(user_data)
            user_prob = self.user_reliability_model.predict_proba(user_features)[0][1] # Probability of being FAKE user

            # Combine: [text, style, tfidf, user_reliability]
            meta_input = np.array([[text_prob, style_prob, tfidf_prob, user_prob]])
            meta_input = self.social_meta_scaler.transform(meta_input)
            
            final_prob = self.social_meta_model.predict_proba(meta_input)[0][1]
            final_pred = self.social_meta_model.predict(meta_input)[0]
            
            user_score = float(user_prob)
        else:
            # Standard website mode
            meta_input = np.array([[text_prob, style_prob, tfidf_prob]])
            meta_input = self.meta_scaler.transform(meta_input)

            final_prob = self.meta_model.predict_proba(meta_input)[0][1]
            final_pred = self.meta_model.predict(meta_input)[0]
            user_score = None

        return {
            "prediction": int(final_pred),
            "label": "FAKE" if final_pred == 1 else "REAL",
            "confidence": float(final_prob),
            "mode": mode,
            "model_scores": {
                "text_model": float(text_prob),
                "style_model": float(style_prob),
                "tfidf_model": float(tfidf_prob),
                "user_reliability": user_score
            }
        }
