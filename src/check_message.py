import joblib
from pathlib import Path

# Load the saved model and vectorizer
project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models"

model_path = model_dir / "spam_model.pkl"
vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

print("Loading Message Shield...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("✓ Model loaded successfully!\n")

def predict_message(message):
    """Predict whether a message is spam or ham"""
    # Vectorize the message
    message_vector = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_vector)[0]
    
    # Get probability scores
    probabilities = model.predict_proba(message_vector)[0]
    
    return prediction, probabilities

print("=" * 70)
print("         MESSAGE SHIELD - SMS Spam Detector")
print("=" * 70)
print("\nType a message to check if it's spam or legitimate.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    # Get user input
    message = input("Enter message: ").strip()
    
    # Check for exit commands
    if message.lower() in ['quit', 'exit', 'q']:
        print("\nThank you for using Message Shield! Goodbye.")
        break
    
    # Skip empty messages
    if not message:
        print("Please enter a message.\n")
        continue
    
    # Make prediction
    prediction, probs = predict_message(message)
    
    # Display results
    print("\n" + "-" * 70)
    if prediction == "spam":
        print("🚨 SPAM DETECTED")
        print(f"   Confidence: {probs[1]:.1%} spam, {probs[0]:.1%} legitimate")
    else:
        print("✓ LEGITIMATE MESSAGE")
        print(f"   Confidence: {probs[0]:.1%} legitimate, {probs[1]:.1%} spam")
    print("-" * 70 + "\n")