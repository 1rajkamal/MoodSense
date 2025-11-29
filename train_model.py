import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# ---------- 1. Small sample dataset (Happy / Sad sentences) ----------

happy_texts = [
    "I am very happy today",
    "I feel great and full of energy",
    "Today is a wonderful day",
    "I am so grateful for my life",
    "Everything is going good",
    "I am excited for the future",
    "My friends make me smile",
    "I got good marks in my exam",
    "I am feeling positive and strong",
    "Aaj mera din bahut accha hai",
    "I am proud of myself",
    "I love spending time with my family",
]

sad_texts = [
    "I am very sad today",
    "I feel lonely and empty inside",
    "Today is a terrible day",
    "Nothing is going right",
    "I am tired and broken",
    "I feel so low and weak",
    "My heart is full of pain",
    "I am disappointed with myself",
    "I feel stressed and tensed",
    "Aaj mera mood bahut kharab hai",
    "I just want to cry",
    "I feel lost and confused",
]

texts = happy_texts + sad_texts
labels = ["happy"] * len(happy_texts) + ["sad"] * len(sad_texts)

# ---------- 2. Make DataFrame ----------

df = pd.DataFrame({"text": texts, "label": labels})
print("Total samples:", len(df))
print(df.head())

# ---------- 3. Train / Test split ----------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ---------- 4. Vectorizer + Model ----------

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ---------- 5. Accuracy check ----------

train_acc = model.score(X_train_vec, y_train)
test_acc = model.score(X_test_vec, y_test)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)

# ---------- 6. Save model + vectorizer ----------

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("mood_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved: vectorizer.pkl and mood_model.pkl in backend folder")
