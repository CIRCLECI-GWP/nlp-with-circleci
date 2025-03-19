import pytest
from app import app  # Import your Flask app from the script where it's defined


@pytest.fixture
def client():
    """Fixture to set up a test client for the Flask app."""
    app.testing = True  # Enable testing mode
    client = app.test_client()
    return client


def test_predict_positive(client):
    """Test the sentiment prediction endpoint with positive input."""
    response = client.post("/predict", json={"text": "I love this movie!"})
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data
    assert data["sentiment"] == "positive"


def test_predict_negative(client):
    """Test the sentiment prediction endpoint with negative input."""
    response = client.post("/predict", json={"text": "This is the worst movie ever!"})
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data
    assert data["sentiment"] == "negative"


def test_no_text_provided(client):
    """Test the endpoint when no text is provided."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "No text provided"
