import nltk
import pandas as pd
import re
import string
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Remove noise from the dataset
def removeNulls(df,name):
    df1 = df[df["review_text"].notnull()]
    df2 = df1[df1["review_text"]!=""]
    df3 = df2.dropna()
    df_clean = df3[df3["INADEQUADA"]==0]
    print(str(name)+":",len(df),str(name)+"_clean:",len(df_clean))
    return df_clean

def removeUpper(df):
    df["review_text"] = df["review_text"].apply(lambda x: x.lower())
    return df

# Remove stopwords from the dataset
def removeStopwords(data):
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    data["review_text"] = data["review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return data

# Remove punctuation from the text
def removePonctuation(df):
    df["review_text"] = df["review_text"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df

def removeNumbers(df):  
    df["review_text"] = df["review_text"].apply(lambda x: re.sub(r'\d+', '', x))
    return df 

def normalizeRepeatedChars(df):
    df["review_text"] = df["review_text"].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1', x))
    return df

def spellChecker(df):
    spell = SpellChecker(language='pt')
    df["review_text"] = df["review_text"].apply(lambda x: ' '.join([spell.correction(word) if word in spell else word for word in x.split()]))
    return df

# Tokenize the text
def tokenize_text(df):
    df["tokens"] = df["review_text"].apply(lambda x: x.split())
    return df

# Lemmatize the text
def lemmatize_text(df):
    lemmatizer = nltk.WordNetLemmatizer()
    df["review_text"] = df["review_text"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return df

# Stem the text
def stem_text(df):
    stemmer = PorterStemmer()
    df["review_text"] = df["review_text"].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    return df

# Generate the corpus
def generate_corpus(df,sw,ps):
    corpus = []
    for i in df.index:
        # get review and remove non alpha chars
        review = re.sub('[^a-zA-Z]', ' ', df['review_text'][i])
        # to lower-case
        review = review.lower()
        # split into tokens, apply stemming and remove stop words
        review = ' '.join([ps.stem(w) for w in review.split() if w not in sw])
        corpus.append(review)
    return corpus

if __name__ == '__main__':


    test = pd.read_csv("data/test.csv") 
    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")

    test_clean = removeNulls(test,"test")
    train_clean = removeNulls(train,"train")
    validation_clean = removeNulls(validation,"validation")

    test_clean = removeUpper(test_clean)
    train_clean = removeUpper(train_clean)
    validation_clean = removeUpper(validation_clean)
    
    #test_clean = removeStopwords(test_clean)
    #train_clean = removeStopwords(train_clean)
    #validation_clean = removeStopwords(validation_clean)

    test_clean = removePonctuation(test_clean) 
    train_clean = removePonctuation(train_clean)
    validation_clean = removePonctuation(validation_clean)

    test_clean = removeNumbers(test_clean)
    train_clean = removeNumbers(train_clean)
    validation_clean = removeNumbers(validation_clean)

    test_clean = normalizeRepeatedChars(test_clean)
    train_clean = normalizeRepeatedChars(train_clean)
    validation_clean = normalizeRepeatedChars(validation_clean)

    test_clean = tokenize_text(test_clean)
    train_clean = tokenize_text(train_clean)
    validation_clean = tokenize_text(validation_clean)

    test_clean = lemmatize_text(test_clean)
    train_clean = lemmatize_text(train_clean)
    validation_clean = lemmatize_text(validation_clean)

    test_clean = stem_text(test_clean)
    train_clean = stem_text(train_clean)
    validation_clean = stem_text(validation_clean)
    
    #test_clean = spellChecker(test_clean)
    #train_clean = spellChecker(train_clean)
    #validation_clean = spellChecker(validation_clean)

    
    test_clean.to_csv("data/test_clean.csv",index=False)
    train_clean.to_csv("data/train_clean.csv",index=False)
    validation_clean.to_csv("data/validation_clean.csv",index=False)

    