import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')

app = st.container()

ps = PorterStemmer()

# function for preprocessing :
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
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


cv = pickle.load(open('vectorizer.pkl', 'rb'))
model= pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    # preprocess :
    transform_sms = transform_text(input_sms)
    # vectorize : 
    vector_input = cv.transform([transform_sms])
    # predict :
    result = model.predict(vector_input)[0]
    # display :
    if result == 1:
        st.header('Spam')
    else :
        st.header('Not Spam')