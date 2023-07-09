#libraries
import pandas as pd
import numpy as np
import re
from sklearn import  metrics
from sklearn.metrics import classification_report
import sys
sys.path.append('../preprocess/')
from text_process import preprocess_text


class KeywordMatcher:
    """
    A simple model that classifies text based on the presence of certain keywords
    """

    def __init__(self, keywords):
        self.keywords = keywords

    def classify_text(self, text):
        """
        This take keywords and match with post kewords and classify
        """
        text = " ".join(text)
        text= preprocess_text(text)
        for keyword in self.keywords['keywords']:
            if re.search(keyword, text, re.IGNORECASE):
                return 1
        return 0

    def predict_proba(self, data):
        """
        Evaluate the model score for each data sample
        """
        predictions = []
        for sen in data:
            predictions.append(self.classify_text(sen))
        return np.array(predictions)


# Read dataset in JSON format and convert to pandas dataframe
test = pd.read_json('../data/dataset.json')


# Step pandas dataframe process label text to neumaric
data = test.T.reset_index().drop(['index'], axis=1)
data['annotators'] = data['annotators'].apply((lambda x: x[0]['label']))
data['annotators'] = data['annotators'].apply((lambda x: re.sub('normal', '0', x)))
data['annotators'] = data['annotators'].apply((lambda x: re.sub('hatespeech', '1', x)))
data['annotators'] = data['annotators'].apply((lambda x: re.sub('offensive', '1', x)))
data['annotators'] = pd.to_numeric(data['annotators'])

# Instantiate the KeywordMatcher model and compute accuracy
keywords = pd.read_csv('../keywords/hatespeech_keywords.csv')
model = KeywordMatcher(keywords)
predictions = model.predict_proba(data['post_tokens'])
accuracy = metrics.classification_report(predictions, data['annotators'])
print('Accuracy of simple hate keyword matching: ', accuracy)
