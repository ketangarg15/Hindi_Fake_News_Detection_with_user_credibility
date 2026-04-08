import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading user datasets...")
real_users = pd.read_csv("data/users.csv")
fake_users = pd.read_csv("data/fusers.csv")

# Label them
real_users['is_fake'] = 0
fake_users['is_fake'] = 1

# Combine
all_users = pd.concat([real_users, fake_users], ignore_index=True)

# Split (80/20)
train_users, test_users = train_test_split(
    all_users, 
    test_size=0.2, 
    random_state=42, 
    stratify=all_users['is_fake']
)

# Save
train_users.to_csv("data/train_users.csv", index=False)
test_users.to_csv("data/test_users.csv", index=False)

print(f"✅ User split created:")
print(f"Train Users: {len(train_users)}")
print(f"Test Users:  {len(test_users)}")
