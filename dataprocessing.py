import nltk
import pandas as pd
import re
import string
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os


#to convert parquet to csv

df_train = pd.read_parquet("data/train.parquet")
df_val = pd.read_parquet("data/validation.parquet")
df_test = pd.read_parquet("data/test.parquet")

df_test.to_csv("data/test.csv", index=False)
df_train.to_csv("data/train.csv", index=False)
df_val.to_csv("data/validation.csv", index=False)

def check_for_nan(data):
    nan_rows = data[data.isna().any(axis=1)]
    if not nan_rows.empty:
        print("Rows with NaN values:")
        print(nan_rows)
    else:
        print("No NaN values found.")
# Remove noise from the dataset
def removeNulls(df,name):
    df1 = df[df["review_text"].notnull()]
    df2 = df1[df1["review_text"]!=""]
    df3 = df2.dropna()
    df_clean = df3[df3["INADEQUADA"]==0]
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

def save_datasets(test, train, validation, folder_name):
    folder_path = os.path.join("data", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    test.to_csv(f"{folder_path}/test_clean.csv", index=False)
    train.to_csv(f"{folder_path}/train_clean.csv", index=False)
    validation.to_csv(f"{folder_path}/validation_clean.csv", index=False)


if __name__ == '__main__':
    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")

    techniques = [
        ("removeUpper", removeUpper),
        ("removeStopwords", removeStopwords),
        ("removePonctuation", removePonctuation),
        ("removeNumbers", removeNumbers),
        ("normalizeRepeatedChars", normalizeRepeatedChars),
        ("tokenize_text", tokenize_text),
        ("lemmatize_text", lemmatize_text),
        ("stem_text", stem_text),
        # ("spellChecker", spellChecker),
    ]

    # Apply only removeNulls to the datasets
    folder_name = "datasets_removeNulls"
    test_clean, train_clean, validation_clean = test.copy(), train.copy(), validation.copy()
    test_clean = removeNulls(test_clean, "test")
    train_clean = removeNulls(train_clean, "train")
    validation_clean = removeNulls(validation_clean, "validation")
    save_datasets(test_clean, train_clean, validation_clean, folder_name)

    # Apply each technique to the datasets
    num_techniques = len(techniques)
    for i in range(0, num_techniques):
        folder_name = "datasets"
        test_clean, train_clean, validation_clean = test.copy(), train.copy(), validation.copy()
        
        # Apply removeNulls before transformations
        test_clean = removeNulls(test_clean, "test")
        train_clean = removeNulls(train_clean, "train")
        validation_clean = removeNulls(validation_clean, "validation")
        
        technique_name, technique_func = techniques[i]
        folder_name += f"_{technique_name}"
        print(f"Applying technique: {technique_name} on dataset {folder_name}")
        test_clean = technique_func(test_clean)
        train_clean = technique_func(train_clean)
        validation_clean = technique_func(validation_clean)
        
        # Apply removeNulls after transformations
        test_clean = removeNulls(test_clean, "test")
        train_clean = removeNulls(train_clean, "train")
        validation_clean = removeNulls(validation_clean, "validation")
        
        save_datasets(test_clean, train_clean, validation_clean, folder_name)


    # Apply all techniques to the datasets
    folder_name = "datasets_all"
    test_clean, train_clean, validation_clean = test.copy(), train.copy(), validation.copy()
    
    # Apply removeNulls before transformations
    test_clean = removeNulls(test_clean, "test")
    train_clean = removeNulls(train_clean, "train")
    validation_clean = removeNulls(validation_clean, "validation")
    
    for technique_name, technique_func in techniques:
        test_clean = technique_func(test_clean)
        train_clean = technique_func(train_clean)
        validation_clean = technique_func(validation_clean)
    
    # Apply removeNulls after transformations
    test_clean = removeNulls(test_clean, "test")
    train_clean = removeNulls(train_clean, "train")
    validation_clean = removeNulls(validation_clean, "validation")
    
    save_datasets(test_clean, train_clean, validation_clean, folder_name)