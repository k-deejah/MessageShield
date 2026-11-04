import sys
import joblib
from pathlib import Path

# Load the saved model and vectorizer
project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models"

model_path = model_dir / "spam_model.pkl"
vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_message(message):
    """Predict whether a message is spam or ham"""
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0]
    return prediction, probabilities

# Get message from command line arguments
if len(sys.argv) < 2:
    print("Usage: python src/classify.py \"Your message here\"")
    print("\nExample:")
    print("  python src/classify.py \"Congratulations! You won a prize!\"")
    sys.exit(1)

# Join all arguments as the message
message = " ".join(sys.argv[1:])

# Make prediction
prediction, probs = predict_message(message)

# Display results
print("\n" + "=" * 70)
print(f"Message: {message}")
print("=" * 70)

if prediction == "spam":
    print("🚨 SPAM DETECTED")
    print(f"   Confidence: {probs[1]:.1%} spam, {probs[0]:.1%} legitimate")
else:
    print("✓ LEGITIMATE MESSAGE")
    print(f"   Confidence: {probs[0]:.1%} legitimate, {probs[1]:.1%} spam")

print("=" * 70 + "\n")