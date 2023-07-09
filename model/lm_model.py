#libraries
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
from abc import ABC

class hateLabels(Dataset, ABC):
    # encoding label,  dataset labels are 0 and 1
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class hateModel(ABC):
    def __init__(self, model_name='bert-base-uncased', lr=1e-5):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()

    def tokenize(self, text):
        return self.tokenizer(text.tolist(), truncation=True, padding=True, max_length=128)

    def create_data_loader(self, encodings, labels, batch_size=16):
        dataset = hateLabels(encodings, labels)
        return DataLoader(dataset, batch_size=batch_size)

    def train(self, train_texts, train_labels, model_name='model.pt', val_texts=None, val_labels=None, epochs=5):
        train_encodings = self.tokenize(train_texts)
        train_loader = self.create_data_loader(train_encodings, train_labels)
        val_loader = None

        if val_texts is not None and val_labels is not None:
            val_encodings = self.tokenize(val_texts)
            val_loader = self.create_data_loader(val_encodings, val_labels)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)

            train_loss = 0
            val_loss = 0

            # Train the model
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs[0]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()

            self.model.eval()

            # Validate the model
            if val_loader:
                for batch in val_loader:
                    with torch.no_grad():
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs[0]
                        val_loss += loss.item()

                print(f'Train loss {train_loss / len(train_loader)}')
                print(f'Validation loss {val_loss / len(val_loader)}')
            else:
                print(f'Train loss {train_loss / len(train_loader)}')

            self.model.train()

        # To save:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,},  model_name)

    # evaluation of model performance
    def evaluate(self, test_texts, test_labels):
      self.model.eval()
      test_encodings = self.tokenize(test_texts)
      test_loader = self.create_data_loader(test_encodings, test_labels)
      predictions = []
      actuals = []

      with torch.no_grad():
          for batch in test_loader:
              batch = {k: v.to(self.device) for k, v in batch.items()}
              outputs = self.model(**batch)
              _, preds = torch.max(outputs.logits, dim=1)
              predictions.extend(preds.tolist())
              actuals.extend(batch['labels'].tolist())

      print(classification_report(actuals, predictions))


# import your data for model 1 (label 0=normal, 1= hate/offensivie)
train_m1 = pd.read_csv('/content/train_m1.csv')
test_m1 = pd.read_csv('/content/test_m1.csv')

# import your data for model 2 (here, label 0=hate, 1= offeinsive)
train_m2 = pd.read_csv('/content/train_m2.csv')
test_m2 = pd.read_csv('/content/test_m2.csv')

my_model = hateModel()
# model_1 train
# if you want to validate during training then pelase load val dataste valid_m1.cs and valid_m2.csv and pass to the train funtion
# e.g., my_model.train(train_m1['review'], train_m1['label'], 'model1.pt', valid_m1['review'], valid_m1['label'])
my_model.train(train_m1['review'], train_m1['label'], 'model1.pt')
my_model.evaluate(test_m1['review'], test_m1['label'])

# model_2 train
my_model.train(train_m2['review'], train_m2['label'], 'model2.pt')
my_model.evaluate(test_m2['review'], test_m2['label'])