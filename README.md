# 📧 Email/SMS Spam Classifier

A simple and efficient **Machine Learning web app** that classifies messages as **Spam 🚫** or **Not Spam ✅** using Natural Language Processing (NLP).

Built with **Streamlit**, **Scikit-learn**, and **NLTK**.

---

## 🚀 Live Demo

*(Add your Streamlit app link here after deployment)*

---

## 📌 Features

* Classifies email/SMS messages into **Spam** or **Not Spam**
* Uses **TF-IDF Vectorization**
* Text preprocessing with:

  * Lowercasing
  * Tokenization
  * Stopword removal
  * Stemming
* Simple and interactive UI using Streamlit

---

## 🧠 Machine Learning Model

* Algorithm: *(e.g., Multinomial Naive Bayes / Logistic Regression — update this)*
* Vectorizer: TF-IDF
* Dataset: *(mention your dataset if any)*

---

## 🗂️ Project Structure

```
Email-Spam-Classifier/
│
├── app.py                # Streamlit app
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Email-Spam-Classifier.git
cd Email-Spam-Classifier
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🧪 How it Works

1. User enters a message
2. Text is preprocessed:

   * Converted to lowercase
   * Tokenized
   * Stopwords removed
   * Stemmed
3. Transformed using TF-IDF vectorizer
4. Model predicts:

   * **Spam** or **Not Spam**

---

## 📦 Requirements

* Python 3.x
* streamlit
* scikit-learn
* nltk

---

## ⚠️ Important Notes

* Make sure `model.pkl` and `vectorizer.pkl` are in the root directory
* NLTK resources (`punkt`, `stopwords`) are automatically downloaded

---

## 🌐 Deployment

This app can be deployed easily using:

* Streamlit Community Cloud

Steps:

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Deploy `app.py`

---

## 📸 Screenshot

*(Add screenshot of your app here)*

---

## 🙌 Acknowledgements

* NLP techniques from NLTK
* UI powered by Streamlit

---

## 📄 License

This project is open-source and free to use.

---

## 👨‍💻 Author

**Your Name**
GitHub: https://github.com/your-username

---

⭐ If you like this project, give it a star!
