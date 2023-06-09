import re
import pickle
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import streamlit as st


def predict_sentiment(text, loaded_model):
    return loaded_model.predict(text)

def split_fn(text):
    return text.split(' ')

def preprocess_text(text):
    stopwords = frozenset(thai_stopwords()) | {"?", ".", ";", ":", "!", '"', "ๆ", "ฯ", "#"}
    remove_whaite_space = text.replace(" ", "")
    remove_emoji = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', remove_whaite_space)
    remove_emoticon = re.sub(r':[a-z_]+:', '', remove_emoji)
    words = word_tokenize(remove_emoticon, engine="newmm")
    filtered_words = [word for word in words if word not in stopwords]
    filtered_words = " ".join(word for word in filtered_words)
    return filtered_words

def main():

    st.title("Sentiment Text Classification ")
    text_input = st.text_input(
        "Enter some text 👇"
    )

    filename = "LinearSVC_model.pickle"
    loaded_model = pickle.load(open(filename, 'rb'))
    filename = 'vectorizer.pickle'
    vectorizer = pickle.load(open(filename, 'rb'))

    texts = text_input
    clean_text = preprocess_text(texts)
    ans = predict_sentiment(vectorizer.transform([clean_text]), loaded_model)
    if text_input:
        st.write("Your text classification is  :   ",  str(ans).replace("[", "").replace(']', '').replace("'", ""))

if __name__ == "__main__":
    main()