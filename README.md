# FakeNewsClassifier
Is it fake or factual? 🕵️‍♂️ Fake News Classifier using Naive Bayes on Kaggle dataset. Implements text vectorization, feature engineering, and model tuning with GridSearchCV to accurately detect misinformation.

## 🔍 Problem Statement

With the rise of misinformation online, identifying fake news automatically has become a critical challenge. This project utilizes the [Kaggle Fake News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) to build an intelligent text classification system.

---

## ⚙️ Pipeline Overview

- **Data Cleaning:** Removal of URLs, punctuation, stopwords, and other noise from news text
- **Feature Engineering:**
  - TF-IDF vectorization for textual features
  - Categorical encoding for metadata like subject and publication day
- **Modeling:**
  - Multinomial Naive Bayes for text classification
  - Hyperparameter tuning via GridSearchCV
- **Evaluation:**
  - Accuracy, confusion matrix, and classification report

---

## 📦 Dependencies

- Python ≥ 3.8  
- `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `scipy`

Install with:
```bash
pip install -r requirements.txt

python fake_news_classifier.py
Or use in a Jupyter notebook: FakeNewsClassifier.ipynb

📈 Results
Achieved ~92% accuracy using optimized alpha parameter for Naive Bayes and TF-IDF features. Stacked bar plots reveal class distributions across subjects and time, helping interpret model decisions.
