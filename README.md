# 🛡️ MessageShield

**AI-Powered SMS Spam Detection System**

MessageShield is a machine learning-based spam detection system that classifies SMS messages as spam or legitimate using Natural Language Processing and Naive Bayes classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.48%25-green.svg)

## ✨ Features

- **High Accuracy**: 98.48% accuracy with 91% spam detection rate
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Real-time Detection**: Instant spam classification with confidence scores
- **Multiple Interfaces**: Web app, CLI, and Python API
- **Dataset Visualization**: Interactive charts and statistics
- **Advanced NLP**: TF-IDF vectorization with bigrams and stop word removal

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.48% |
| **Spam Detection Rate** | 91% |
| **False Positives** | 3 out of 1,115 |
| **False Negatives** | 14 out of 1,115 |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/MessageShield.git
cd MessageShield
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - pre-trained model included):
```bash
python src/train_improved.py
```

### Usage

#### Web Interface (Recommended)

Launch the interactive web app:
```bash
python -m streamlit run app.py
```

Or double-click `run_app.bat` (Windows)

The app will open at `http://localhost:8501`

#### Command Line

Check a single message:
```bash
python src/classify.py "Your message here"
```

Example:
```bash
python src/classify.py "Congratulations! You won $1000. Click here now!"
```

#### Interactive CLI

For multiple checks:
```bash
python src/check_message.py
```

## 📁 Project Structure

```
MessageShield/
├── app.py                      # Streamlit web application
├── data/
│   └── SMSSpamCollection      # Dataset (5,572 messages)
├── models/
│   ├── spam_model.pkl         # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── src/
│   ├── main.py                # Original training script
│   ├── train_improved.py      # Improved model training
│   ├── classify.py            # CLI message classifier
│   ├── check_message.py       # Interactive checker
│   └── predict.py             # Batch prediction script
├── requirements.txt           # Python dependencies
├── run_app.bat               # Windows launcher
└── README.md                 # This file
```

## 🧠 Technical Details

### Model Architecture

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 5,000 most important features
- **N-grams**: Unigrams and Bigrams (1-2 word phrases)
- **Preprocessing**: Lowercase conversion, English stop words removal

### Dataset

- **Source**: SMS Spam Collection Dataset
- **Total Messages**: 5,572
  - Legitimate (Ham): 4,825 (86.6%)
  - Spam: 747 (13.4%)
- **Split**: 80% training (4,457), 20% testing (1,115)
- **Stratified**: Maintains class distribution in both sets

### Key Improvements

1. **Bigram Analysis**: Captures phrases like "click here", "win prize"
2. **Stop Word Removal**: Filters common words for better signal
3. **Tuned Alpha Parameter** (0.1): Increased sensitivity to spam indicators
4. **Optimized Vocabulary**: 5,000 carefully selected features

## 📸 Screenshots

### Web Interface - Spam Detection
![Spam Detection](https://via.placeholder.com/800x400?text=Spam+Detection+Interface)

### Dataset Dashboard
![Dashboard](https://via.placeholder.com/800x400?text=Dataset+Statistics+Dashboard)

## 🛠️ Development

### Training a New Model

```bash
python src/train_improved.py
```

This will:
1. Load the SMS dataset
2. Preprocess and vectorize text
3. Train the Naive Bayes classifier
4. Evaluate performance
5. Save model to `models/` directory

### Testing

Test the model with sample messages:
```bash
python src/predict.py
```

## 📈 Future Improvements

- [ ] Add support for MMS and email spam detection
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Create REST API endpoint
- [ ] Add multi-language support
- [ ] Implement real-time learning from user feedback
- [ ] Mobile app integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SMS Spam Collection Dataset from UCI Machine Learning Repository
- Scikit-learn for machine learning tools
- Streamlit for the amazing web framework

## 📧 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/YOUR_USERNAME/MessageShield](https://github.com/YOUR_USERNAME/MessageShield)

---

⭐ Star this repository if you find it helpful!