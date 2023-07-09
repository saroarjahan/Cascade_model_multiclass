import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # convert to lowercase
    text = text.lower()
    
    # remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # tokenize text
    tokens = text.split()
    
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # join tokens back into string
    text = ' '.join(tokens)
    
    return text