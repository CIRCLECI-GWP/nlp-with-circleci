import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Path to the saved model
MODEL_NAME = "ajikadev/circleci-nlp-model"
# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()


def predict_sentiment(text):
    """Tokenizes input text and returns a sentiment prediction."""
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment


# Test the inference function
if __name__ == "__main__":
    text = "I absolutely loved this movie! It was fantastic."
    result = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result}")
