from model_pipeline import FakeNewsPipeline

pipeline = FakeNewsPipeline()

if __name__ == "__main__":
    text = input("Enter article:\n")

    result = pipeline.predict(text)

    print("\nPrediction:", result["label"])
    print(f"Meta Model: {result['confidence']:.4f}")

    print("\n--- Model Breakdown ---")
    print(f"Text Model:  {result['model_scores']['text_model']:.4f}")
    print(f"Style Model: {result['model_scores']['style_model']:.4f}")
    print(f"TF-IDF Model:{result['model_scores']['tfidf_model']:.4f}")