# 🐦 TWITTER-SENTIMENT-ANALYSIS USING MACHINE LEARNING

A machine learning project that analyzes sentiment from tweets to classify them as **positive**, **negative**, or **neutral**. This repository demonstrates the end-to-end pipeline of data collection, preprocessing, model training, evaluation, and visualization using Python.

# 📌 Project Overview

Social media platforms like Twitter are rich sources of public opinion. This project leverages machine learning techniques to extract sentiment from tweets, enabling businesses and researchers to understand public mood and trends in real time.

# 🚀 Features

- Scrapes tweets using Twitter API or preloaded datasets
- Cleans and preprocesses text data (stopwords removal, tokenization, stemming)
- Converts text to numerical format using TF-IDF or CountVectorizer
- Trains multiple ML models (Logistic Regression, Naive Bayes, SVM)
- Evaluates model performance using accuracy, precision, recall, F1-score
- Visualizes sentiment distribution and word clouds

# 🧠 Machine Learning Models Used

| Model              | Description                                  |
|--------------------|----------------------------------------------|
| Logistic Regression| Baseline classifier for binary/multi-class   |
| Naive Bayes        | Probabilistic model suitable for text data   |
| SVM                | High-performance classifier for large feature spaces |

# 📊 Dataset

- Source: [Sentiment140](http://help.sentiment140.com/for-students) or Twitter API
- Format: CSV with tweet text and sentiment labels
- Size: ~1.6 million tweets

# 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, VS Code
- **Visualization**: WordCloud, Confusion Matrix, Bar Charts
