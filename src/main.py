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

# Display the first five messages 
print(df.head())

# Check the dataset dimensions
print("\nDataset shape:", df.shape)

# Count spam and legitimate messages
print("\nMessage counts:")
print(df["label"].value_counts())

# Total messages
print("\nTotal messages:", len(df))

# Percentage distribution
print("\nPercentage distribution:")
print(df["label"].value_counts(normalize=True) * 100)

# Separate features and labels
X = df["message"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Vectorize the text data using TF-IDF
print("\n--- Text Vectorization ---")
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

print("Training data shape:", X_train_vectors.shape)
print("Testing data shape:", X_test_vectors.shape)

# Train the Naive Bayes model
print("\n--- Training Model ---")
model = MultinomialNB()
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
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
print("\n--- Saving Model ---")
project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models"

# Create models directory if it doesn't exist
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "spam_model.pkl"
vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")