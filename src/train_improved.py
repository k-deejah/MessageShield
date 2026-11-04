import pandas as pd 
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Locate our dataset
data_path = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection"

# Load the dataset
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset shape:", df.shape)
print("\nMessage counts:")
print(df["label"].value_counts())

# Separate features and labels
X = df["message"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Vectorize with improved parameters
print("\n--- Text Vectorization (Improved) ---")
vectorizer = TfidfVectorizer(
    max_features=5000,      # Increase vocabulary size
    ngram_range=(1, 2),     # Use unigrams and bigrams (captures phrases like "click here", "win prize")
    min_df=2,               # Ignore very rare words
    max_df=0.8,             # Ignore very common words
    lowercase=True,
    stop_words='english'    # Remove common English words
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

print("Training data shape:", X_train_vectors.shape)
print("Testing data shape:", X_test_vectors.shape)

# Train with adjusted parameters
print("\n--- Training Improved Model ---")
model = MultinomialNB(alpha=0.1)  # Lower alpha = more sensitive to individual words
model.fit(X_train_vectors, y_train)
print("Model trained successfully!")

# Make predictions
print("\n--- Making Predictions ---")
y_pred = model.predict(X_test_vectors)

# Evaluate the model
print("\n--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nFalse Positives (ham marked as spam): {cm[0][1]}")
print(f"False Negatives (spam marked as ham): {cm[1][0]}")

# Save the improved model
print("\n--- Saving Improved Model ---")
project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models"
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "spam_model.pkl"
vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

# Test specific problematic messages
print("\n--- Testing Problematic Messages ---")
test_messages = [
    "hey, click this link you win 400$",
    "Congratulations! Click here to claim your prize",
    "FREE entry to win cash now!",
    "Hey, are we meeting tomorrow?",
]

for msg in test_messages:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    print(f"\nMessage: {msg[:50]}...")
    print(f"  Prediction: {pred.upper()} (spam: {probs[1]*100:.1f}%, ham: {probs[0]*100:.1f}%)")