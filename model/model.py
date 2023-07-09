#libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set max sequence length
MAX_SEQ_LENGTH = 128

class Model:
    def load_model(self, load_path):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f'Model loaded from <== {load_path}')
        return model

    # predict sentence label , for model 1, (0 prediction refers to normal, 1 refers to hate/offensive), 
    # for model 2 (0 prediction refers to hate, 1 refers to offensive), during training I have made training dataset like that
  
    def predict_hate(self, model, sentence):
        tokens = tokenizer.encode_plus(
            sentence,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt')
        tokens = tokens.to(device)
        with torch.no_grad():
            outputs = model(tokens['input_ids'], token_type_ids=None, attention_mask=tokens['attention_mask'])
        logits = outputs[0]
        _, predicted = torch.max(logits, dim=1)
        return predicted.item()

    def predict_proba(self, data):
    # Load Model and Evaluate, final out put would be (0 prediction refers to normal, 1 refers to hate and 2 refers to offensive)
        model1 = self.load_model('model_1.pt')
        model2 = self.load_model('model_2.pt')

        predictions=[]
        for post in data:
            result1=self.predict_hate(model1, post)
            if result1==0:
                predictions.append(0)
            else:
                result2=self.predict_hate(model2, post)
                if result2==0:
                    predictions.append(1)
                else:
                    predictions.append(2)
        return np.array(predictions)

# Instantiate the model
model = Model()

# Read dataset in CSV format and convert to pandas dataframe
test = pd.read_csv('../splited_data/test3.csv')

predictions = model.predict_proba(test['review'][:1000])

# Get user input for classification
accuracy = metrics.classification_report(test['label'][:1000], predictions, digits=3)
print('Accuracy of model cascade: \n')
print(accuracy)
