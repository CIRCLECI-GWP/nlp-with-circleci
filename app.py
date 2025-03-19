from flask import Flask, request, jsonify
from inference import predict_sentiment  # Import the function

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        sentiment = predict_sentiment(text)
        return jsonify({"text": text, "sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
