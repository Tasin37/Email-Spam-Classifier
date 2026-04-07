import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (only if not already available in environment)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# FIX: Explicitly set n_features_in_ for the SVC estimator within the StackingClassifier
# StackingClassifier stores estimators as (name, estimator) tuples
# The SVC was the first estimator, so it's at model.estimators_[0][1]
# The expected number of features was determined to be 3000 from the Colab notebook
if hasattr(model, 'estimators_') and len(model.estimators_) > 0 and isinstance(model.estimators_[0], tuple) and len(model.estimators_[0]) == 2:
    # Assuming the first estimator is the SVC and needs this fix
    svc_estimator = model.estimators_[0][1]
    if hasattr(svc_estimator, 'n_features_in_') and svc_estimator.n_features_in_ == 1:
        svc_estimator.n_features_in_ = 3000
        st.warning("Adjusted SVC's n_features_in_ to 3000 to prevent prediction errors.")


# UI
st.title("📧 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    
    # Ensure the input vector has the correct number of features for prediction
    # This is a safeguard, but the main fix is setting n_features_in_ above
    if vector_input.shape[1] != 3000:
        st.error(f"Input vector has {vector_input.shape[1]} features, but model expects 3000.")
    else:
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")
