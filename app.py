import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
model = joblib.load(BASE_DIR / "models" / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="MessageShield",
    page_icon="🛡️",
    layout="wide"
)

# Application heading
st.title("🛡️ MessageShield")
st.subheader("AI-Powered SMS Spam Detection")

# Create tabs
tab1, tab2 = st.tabs(["🔍 Spam Detector", "📊 Dataset Overview"])

# Tab 1: Spam Detection
with tab1:
    st.write(
        "Protect yourself from suspicious messages. "
        "Enter an SMS below to check whether it is spam or legitimate."
    )
    
    # Message input
    message = st.text_area("Enter your SMS message:", height=100)
    
    # Prediction button
    if st.button("Analyze Message", type="primary"):
        if message.strip():
            # Vectorize and predict
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]
            probabilities = model.predict_proba(message_vector)[0]
            
            # Calculate confidence
            confidence = max(probabilities) * 100
            
            # Display results
            st.divider()
            
            if prediction == "spam":
                st.error("⚠️ Warning: This message is classified as SPAM.")
            else:
                st.success("✅ This message is classified as HAM (Legitimate).")
            
            # Show confidence metric
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
            
            # Show detailed probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Probability", f"{probabilities[0]*100:.2f}%")
            with col2:
                st.metric("Spam Probability", f"{probabilities[1]*100:.2f}%")
        else:
            st.warning("Please enter a message first.")

# Tab 2: Dataset Overview
with tab2:
    st.markdown("### Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", "5,572")
    
    with col2:
        st.metric("Legitimate Messages", "4,825")
    
    with col3:
        st.metric("Spam Messages", "747")
    
    with col4:
        st.metric("Model Accuracy", "98.48%")
    
    st.divider()
    
    # Message Distribution Chart
    st.markdown("### Message Distribution")
    
    chart_data = pd.DataFrame({
        "Message Type": ["Legitimate (Ham)", "Spam"],
        "Count": [4825, 747]
    })
    
    st.bar_chart(chart_data, x="Message Type", y="Count", use_container_width=True)
    
    st.divider()
    
    # Model Performance Metrics
    st.markdown("### Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Spam Detection Rate", "91%", help="Percentage of spam messages correctly identified")
    
    with col2:
        st.metric("False Positives", "3", help="Legitimate messages incorrectly marked as spam")
    
    with col3:
        st.metric("False Negatives", "14", help="Spam messages incorrectly marked as legitimate")
    
    st.divider()
    
    # Technical Details
    with st.expander("📋 Technical Details"):
        st.markdown("""
        **Model:** Multinomial Naive Bayes
        
        **Features:**
        - TF-IDF Vectorization with 5,000 features
        - N-grams: Unigrams and Bigrams (1-2 words)
        - Stop words removed
        - Alpha parameter: 0.1 (increased sensitivity)
        
        **Dataset:** SMS Spam Collection
        - Training set: 4,457 messages (80%)
        - Test set: 1,115 messages (20%)
        - Stratified split to maintain class distribution
        """)

# Footer
st.divider()
st.caption("Powered by Machine Learning | Naive Bayes Classifier with TF-IDF")