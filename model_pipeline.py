import joblib
import numpy as np
class FakeNewsPipeline:

    def __init__(self, model_dir="models/"):
        print("Loading pipeline models...")

        self.text_clf = joblib.load(model_dir + "text_model.pkl")
        self.style_clf = joblib.load(model_dir + "style_model.pkl")
        self.tfidf_clf = joblib.load(model_dir + "tfidf_model.pkl")
        self.tfidf_vectorizer = joblib.load(model_dir + "tfidf_vectorizer.pkl")
        self.meta_model = joblib.load(model_dir + "meta_model.pkl")
        self.embedder = joblib.load(model_dir + "embedding_model.pkl")
        self.scaler = joblib.load(model_dir + "meta_scaler.pkl")

        print(" Pipeline ready!")

    def extract_style_features(self, text):
        import re
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

        hindi_stopwords = [
            "है","हैं","था","थे","की","का","के","को","में",
            "पर","से","तक","और","या","लेकिन","अगर","तो",
            "ही","भी","नहीं","क्या","क्यों","कौन","कब",
            "यह","वह","इन","उन","उस","इस","कुछ","सब",
            "बहुत","अब","जब","तब","जहाँ","वहाँ"
        ]

        stopword_count = sum(1 for w in words if w in hindi_stopwords)
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

    def predict(self, text):

        text = text.lower()

        embedding = self.embedder.encode([text])
        style_features = self.extract_style_features(text)
        tfidf_features = self.tfidf_vectorizer.transform([text])

        text_prob = self.text_clf.predict_proba(embedding)[0][1]
        style_prob = self.style_clf.predict_proba(style_features)[0][1]
        tfidf_prob = self.tfidf_clf.predict_proba(tfidf_features)[0][1]

        meta_input = np.array([[text_prob, style_prob, tfidf_prob]])

        # APPLY SCALING
        meta_input = self.scaler.transform(meta_input)

        final_prob = self.meta_model.predict_proba(meta_input)[0][1]
        final_pred = self.meta_model.predict(meta_input)[0]

        return {
            "prediction": int(final_pred),
            "label": "FAKE" if final_pred == 1 else "REAL",
            "confidence": float(final_prob),
            "model_scores": {
                "text_model": float(text_prob),
                "style_model": float(style_prob),
                "tfidf_model": float(tfidf_prob)
            }
        }