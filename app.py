from flask import Flask, render_template, request
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

_vader = None
_vader_ready = False


def _ensure_vader():
    """Load VADER once. Download lexicon if missing, but fail gracefully offline."""
    global _vader, _vader_ready
    if _vader_ready:
        return _vader
    try:
        _vader = SentimentIntensityAnalyzer()
        _vader_ready = True
        return _vader
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            _vader = SentimentIntensityAnalyzer()
            _vader_ready = True
            return _vader
        except Exception:
            _vader_ready = True
            return None


def _keyword_score(text_lower, keyword_groups):
    score = 0
    matches = []
    for keyword, weight in keyword_groups.items():
        if keyword in text_lower:
            score += weight
            matches.append(keyword)
    return score, matches


def analyze_vibe(text):
    text = (text or "").strip()
    if not text:
        return {
            "mood": "Neutral \U0001F60C",
            "vibe": "Awaiting Input",
            "response": "Type a few lines about how you feel.",
            "advice": "Even short messages help the model understand your vibe.",
            "confidence": 0.0,
            "reasoning": "No input text provided.",
        }

    text_lower = text.lower()

    health_keywords = {
        "cramps": 4,
        "period": 3,
        "vomit": 4,
        "throwing up": 4,
        "nausea": 4,
        "nauseous": 4,
        "pain": 3,
        "chills": 3,
        "fever": 4,
        "headache": 3,
        "sick": 3,
    }
    health_score, health_matches = _keyword_score(text_lower, health_keywords)
    if health_score >= 4:
        return {
            "mood": "Low Physical Energy \U0001F912",
            "vibe": "Recovery Mode \U0001FAE7",
            "response": "Your message signals physical discomfort.",
            "advice": "Hydrate, rest, and seek medical support if symptoms worsen.",
            "confidence": min(0.98, 0.65 + (health_score / 20)),
            "reasoning": f"Health keywords detected: {', '.join(health_matches)}",
        }

    vibe_keywords = {
        "magnetic": (
            "Magnetic \U0001F9F2",
            "Attractive Energy",
            "You are drawing attention naturally.",
            "Channel this into meaningful action.",
        ),
        "confident": (
            "Confident \U0001F525",
            "Boss Energy",
            "You sound clear and self-assured.",
            "Use this momentum for hard tasks.",
        ),
        "tired": (
            "Low \U0001F614",
            "Burnout Vibe",
            "Your energy sounds depleted.",
            "Resting now can improve tomorrow's focus.",
        ),
        "happy": (
            "Happy \u2728",
            "Main Character Energy",
            "You sound positive and bright.",
            "Capture this moment and keep moving.",
        ),
        "overthinking": (
            "Anxious \U0001F9E0",
            "Overthinking Spiral",
            "Your thoughts sound crowded.",
            "Slow the pace and prioritize one step.",
        ),
        "romantic": (
            "Romantic \U0001F496",
            "Lover Energy",
            "You sound emotionally open.",
            "Stay grounded while feeling deeply.",
        ),
        "delusional": (
            "Dreamer \U0001F319",
            "Dreamer Energy",
            "You are thinking beyond limits.",
            "Pair belief with practical action.",
        ),
    }

    for keyword, (mood, vibe, response, advice) in vibe_keywords.items():
        if keyword in text_lower:
            return {
                "mood": mood,
                "vibe": vibe,
                "response": response,
                "advice": advice,
                "confidence": 0.9,
                "reasoning": f"Matched direct vibe keyword: {keyword}",
            }

    vader = _ensure_vader()
    vader_compound = vader.polarity_scores(text)["compound"] if vader else 0.0
    textblob_polarity = TextBlob(text).sentiment.polarity

    positive_lexicon = {
        "excited": 2,
        "grateful": 2,
        "calm": 1,
        "motivated": 2,
        "proud": 2,
        "peaceful": 2,
    }
    negative_lexicon = {
        "sad": 2,
        "angry": 2,
        "stressed": 2,
        "anxious": 2,
        "lonely": 2,
        "hopeless": 3,
        "worried": 2,
    }

    pos_score, pos_matches = _keyword_score(text_lower, positive_lexicon)
    neg_score, neg_matches = _keyword_score(text_lower, negative_lexicon)
    emotion_score = (pos_score - neg_score) / 6.0

    combined = (0.45 * vader_compound) + (0.4 * textblob_polarity) + (0.15 * emotion_score)
    confidence = min(0.95, max(abs(combined), abs(vader_compound), abs(textblob_polarity)) + 0.25)

    if combined > 0.35:
        mood, vibe = "Happy \u2728", "Positive Energy \U0001F4AB"
        response, advice = (
            "Things seem to be moving in a good direction.",
            "Build on this with one concrete next win.",
        )
    elif combined < -0.35:
        mood, vibe = "Low \U0001F614", "Negative Energy \U0001F327\uFE0F"
        response, advice = (
            "You sound emotionally heavy right now.",
            "Take one gentle step: breathe, pause, then reset.",
        )
    else:
        mood, vibe = "Neutral \U0001F60C", "Balanced \U0001F33F"
        response, advice = (
            "Your emotional signal looks mixed or steady.",
            "A small plan for today can create clarity.",
        )

    parts = [
        f"VADER={vader_compound:.2f}",
        f"TextBlob={textblob_polarity:.2f}",
        f"EmotionLex={emotion_score:.2f}",
    ]
    if pos_matches:
        parts.append(f"positive words: {', '.join(pos_matches)}")
    if neg_matches:
        parts.append(f"negative words: {', '.join(neg_matches)}")

    return {
        "mood": mood,
        "vibe": vibe,
        "response": response,
        "advice": advice,
        "confidence": round(confidence, 2),
        "reasoning": "; ".join(parts),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("text", "")
        result = analyze_vibe(text)
        result["text"] = text

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)