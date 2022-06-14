# libraries
import joblib
import pickle
import spacy
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Data for news trained dataset
# use for tokeinzation and remove stopwords from dataset
nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)
punct = string.punctuation


app = Flask(__name__)


def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens


# from sklearn.naive_bayes import MultinomialNB
# # model = numpy.load('data.npz')
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)

app.secret_key = "my precious"


@app.route('/')
# @login_required
def home():
    Cat = ""

    # render a template
    return render_template('NewsClassify.html', Category=Cat)
    # return "Hello, World!"  # return a string


@app.route('/classify', methods=['POST', 'GET'])
def Classfy():

    if request.method == 'POST':
        model = joblib.load('news_classifier.pkl')
        text = request.form['Name']

        pred = model.predict([text])
        if pred == 'b':
            predA = "Business News"
        elif pred == 't':
            predA = "Science and Technology"
        elif pred == 'e':
            predA = "Entertainment"
        elif pred == 'm':
            predA = "Health"

        s = predA
        # a = accuracy_score(model,pred)

        return render_template('NewsClassify.html', Category=s)
    else:
        return render_template('NewsClassify.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
