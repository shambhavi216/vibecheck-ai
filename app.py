from textblob import TextBlob
from flask import Flask, render_template, request
import nltk
nltk.download('punkt')

app = Flask(__name__)


def analyze_vibe(text):
    text_lower = text.lower()

    # 💊 Health / period / discomfort detection
    health_keywords = ["cramps", "period", "vomit",
                       "throwing up", "nausea", "nauseous", "pain", "chills"]

    for word in health_keywords:
        if word in text_lower:
            return (
                "Unwell 🤍",
                "Low Physical Energy",
                "Your body is asking for rest.",
                "Drink water, rest well, and take care. If it gets worse, consider medical help."
            )

    # 💖 Emotion/personality keywords
    vibe_dict = {
        "magnetic": ("Magnetic 🧲", "Attractive Energy", "You're pulling everything towards you.", "Your presence is powerful."),
        "confident": ("Confident 🔥", "Boss Energy", "You know your worth.", "Keep owning your space."),
        "tired": ("Low 😔", "Burnout Vibe", "You're drained.", "Rest is productive too."),
        "happy": ("Happy ✨", "Main Character Energy 💅", "You're glowing.", "Enjoy this energy."),
        "overthinking": ("Anxious 🧠", "Overthinking Spiral", "Your mind is racing.", "Slow down your thoughts."),
        "romantic": ("Romantic 💕", "Lover Energy", "You're feeling soft.", "Feel it deeply."),
        "delusional": ("Delulu 🌙", "Dreamer Energy", "It might actually work.", "Take action with belief."),
    }

    for word in vibe_dict:
        if word in text_lower:
            return vibe_dict[word]

    # 🤖 fallback AI
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.4:
        return ("Happy ✨", "Positive Energy", "Things are going well.", "Keep going.")
    elif polarity < -0.4:
        return ("Low 😔", "Negative Energy", "You're feeling down.", "Take it slow.")
    else:
        return ("Neutral 😌", "Balanced", "You're steady.", "Stay grounded.")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["text"]
        mood, vibe, response, advice = analyze_vibe(text)

        result = {
            "text": text,
            "mood": mood,
            "vibe": vibe,
            "response": response,
            "advice": advice
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
