# CREATE TRAIN / TEST SPLIT

import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading datasets...")

fake_df = pd.read_csv("data/hindi_fake_news.csv")
true_df = pd.read_csv("data/hindi_true_news.csv")

# Assign labels
fake_df['label'] = 1
true_df['label'] = 0

# Combine
news_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create content column
news_df['content'] = news_df['Articles'].fillna('').str.lower()

# Split
train_df, test_df = train_test_split(
    news_df,
    test_size=0.2,
    random_state=42,
    stratify=news_df['label']
)

# Save
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Dataset created:")
print("Train:", train_df.shape)
print("Test:", test_df.shape)