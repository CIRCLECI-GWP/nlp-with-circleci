import pytest
from inference import predict_sentiment  # Import the function to test


def test_predict_positive():
    """Test if the function correctly predicts a positive sentiment."""
    result = predict_sentiment("I absolutely love this movie! It was fantastic.")
    assert result == "positive"


def test_predict_negative():
    """Test if the function correctly predicts a negative sentiment."""
    result = predict_sentiment("This is the worst movie I have ever watched.")
    assert result == "negative"


def test_neutral_sentiment():
    """Test with a neutral sentence."""
    result = predict_sentiment("The movie was okay, nothing special.")
    assert result in ["positive", "negative"]  # Depending on how the model was trained


def test_empty_string():
    """Test with an empty string."""
    result = predict_sentiment("")
    assert result in ["positive", "negative"]  # Model might still classify it


def test_non_english_text():
    """Test with a non-English sentence."""
    result = predict_sentiment("Je t'aime beaucoup, c'était magnifique.")  # French
    assert result in ["positive", "negative"]  # Model may struggle with non-English
