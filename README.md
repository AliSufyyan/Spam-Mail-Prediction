# 📧 Spam Mail Prediction System

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using NLP techniques — TF-IDF vectorization and Logistic Regression.

---

## 📌 Project Overview

This project builds a complete text classification pipeline: raw SMS messages are converted into numerical TF-IDF feature vectors and fed into a Logistic Regression classifier. The trained model achieves **96.47% test accuracy** with virtually no gap from training accuracy — indicating strong generalization.

This is the first NLP-based project in the portfolio, introducing the full text-to-prediction pipeline.

---

## 📊 Dataset

**File:** `mail_data.csv`  
**Source:** SMS Spam Collection Dataset — UCI Machine Learning Repository

| Property | Value |
|---|---|
| Total Messages | 5,572 |
| Ham (Not Spam) | 4,825 messages (86.6%) |
| Spam | 747 messages (13.4%) |
| Train / Test Split | 70% / 30% |
| Vocabulary Size | 6,896 unique terms (after stop word removal) |

---

## 📈 Results

| Metric | Score |
|---|---|
| Training Accuracy | 96.62% |
| Test Accuracy | 96.47% |

The gap between training and test accuracy is only **0.15%** — the model generalizes exceptionally well with virtually no overfitting.

---

## 🔍 How It Works

```
Raw SMS Text
     ↓
Remove stop words + lowercase (TfidfVectorizer)
     ↓
Convert to TF-IDF numerical vectors (6,896 features)
     ↓
Logistic Regression classifier
     ↓
Spam or Ham
```

**Critical rule:** `fit_transform()` is applied only to training data. Test data uses `transform()` only — ensuring the same vocabulary is used throughout.

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy
- Pandas
- Scikit-learn (TfidfVectorizer, LogisticRegression, train_test_split, accuracy_score)

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AliSufyaan/spam-mail-prediction.git
   cd spam-mail-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook Spam_Mail_Prediction.ipynb
   ```

---

## 📁 Project Structure

```
spam-mail-prediction/
├── Spam_Mail_Prediction.ipynb    # Main notebook
├── mail_data.csv                 # Dataset
└── README.md                     # You are here
```

---

## ⚠️ Limitations

- **Class imbalance** — 86.6% ham vs 13.4% spam means accuracy alone is misleading; a model predicting everything as ham would score 86% without learning anything useful
- **No confusion matrix** — precision, recall and F1-score not yet reported; false negatives (missed spam) matter more than raw accuracy for this use case
- **No class balancing applied** — `class_weight='balanced'` not used in LogisticRegression

---

## 🧠 What I Learned

- Full NLP text classification pipeline from raw text to predictions
- How TF-IDF converts text into meaningful numerical features
- Why `fit_transform()` must only be used on training data and `transform()` on test data
- How class imbalance affects accuracy metrics and why precision/recall matter more for spam detection

---

## 📚 Reference

- Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Also available on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
