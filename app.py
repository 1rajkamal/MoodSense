from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# --------- Load saved model & vectorizer ----------
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("mood_model.pkl", "rb") as f:
    model = pickle.load(f)


# --------- Route ----------
@app.route("/", methods=["GET", "POST"])
def home():
    mood = None
    confidence = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("message", "").strip()

        if user_text:
            x_vec = vectorizer.transform([user_text])
            pred = model.predict(x_vec)[0]
            probs = model.predict_proba(x_vec)[0]

            conf = probs.max() * 100
            confidence = f"{conf:.2f}"

            if pred == "happy":
                mood = "HAPPY 😁"
            else:
                mood = "SAD 😢"
        else:
            mood = "Please type something 🙂"

    return render_template("index.html", mood=mood, confidence=confidence, user_text=user_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
