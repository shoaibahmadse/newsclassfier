from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.naive_bayes import MultinomialNB
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
import pandas as pd
import numpy as np
from sklearn import naive_bayes, metrics, model_selection, preprocessing

news_data = pd.read_csv('uci-news-aggregator.csv',
                        nrows=10000, error_bad_lines=False)
punct = string.punctuation

# dataCleaning
# import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)


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

# print(text_data_cleaning("  tis is the best in the salma"))


# classification
# #     # split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
    news_data['TITLE'], news_data['CATEGORY'], test_size=0.2, random_state=0)
X = news_data['TITLE']
y = news_data['CATEGORY']
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
classifier = MultinomialNB()


text = input(
    'enter news text relevant to business,science,entertainment, health: \n')

clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(train_x, train_y)
pred = clf.predict([text])
if pred == 'b':
    print("Business News")
elif pred == 't':
    print("Science and Technology")
elif pred == 'e':
    print("Entertainment")
elif pred == 'm':
    print("Health")

print("Accuracy: \n", accuracy_score(valid_y, clf.predict(valid_x)))

print(classification_report(valid_y, clf.predict(valid_x)))
joblib.dump(clf, 'news_classifier.pkl')
