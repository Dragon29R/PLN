import nltk
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import re

# Remove noise from the dataset
def removeNulls(df,name):
    df1 = df[df["review_text"].notnull()]
    df2 = df1[df1["review_text"]!=""]
    df3 = df2.dropna()
    df_clean = df3[df3["INADEQUADA"]==0]
    print(str(name)+":",len(df),str(name)+"_clean:",len(df_clean))
    return df_clean

# Remove stopwords from the dataset
def remove_stopwords(data):
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    data["review_text"] = data["review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

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

    test_clean = remove_noise(test,"test")
    train_clean = remove_noise(train,"train")
    validation_clean = remove_noise(validation,"validation")

    remove_stopwords(test_clean)
    remove_stopwords(train_clean)
    remove_stopwords(validation_clean)

    test_clean.to_csv("data/test_clean.csv",index=False)
    train_clean.to_csv("data/train_clean.csv",index=False)
    validation_clean.to_csv("data/validation_clean.csv",index=False)

