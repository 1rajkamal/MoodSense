import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 1. CSV read karo (jo maine diya hai)
df = pd.read_csv("mood_kaggle.csv")

print("Columns in CSV:", df.columns)
print("First 5 rows:")
print(df.head())

# Columns ka naam fix hai: text + label (happy / sad)
TEXT_COL = "text"
LABEL_COL = "label"

df = df[[TEXT_COL, LABEL_COL]].dropna()

# labels already "happy"/"sad" hain, just lowercase safety ke liye
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower()
df = df[df[LABEL_COL].isin(["happy", "sad"])]
df["label"] = df[LABEL_COL]

print("\nLabel counts:")
print(df["label"].value_counts())

X = df[TEXT_COL]
y = df["label"]

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF + Model
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# 4. Evaluation
print("\nTrain accuracy:", model.score(X_train_vec, y_train))
print("Test accuracy:", model.score(X_test_vec, y_test))

y_pred = model.predict(X_test_vec)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Save model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("mood_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ New model saved: vectorizer.pkl & mood_model.pkl")
print("   Flask app ab isi trained model ko use karega.")
