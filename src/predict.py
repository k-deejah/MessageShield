import joblib
from pathlib import Path

# Load the saved model and vectorizer
project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models"

model_path = model_dir / "spam_model.pkl"
vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

print("Loading model...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("Model loaded successfully!\n")

def predict_message(message):
    """Predict whether a message is spam or ham"""
    # Vectorize the message
    message_vector = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_vector)[0]
    
    # Get probability scores
    probabilities = model.predict_proba(message_vector)[0]
    
    return prediction, probabilities

# Test with some example messages
test_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Verify your identity immediately!",
    "Thanks for the notes from class today. See you next week!",
    "FREE entry to win cash prizes. Text WIN to 12345 now!",
]

print("--- Spam Detection Results ---\n")
for i, message in enumerate(test_messages, 1):
    prediction, probs = predict_message(message)
    
    print(f"Message {i}: {message[:60]}...")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: ham={probs[0]:.2%}, spam={probs[1]:.2%}")
    print("-" * 70)