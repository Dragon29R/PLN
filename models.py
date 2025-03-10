import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, RocCurveDisplay

import pandas as pd

def nearest_neighbours(train_x,train_y,test_x,test_y, n_neighbours=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbours)
    clf.fit(train_x,train_y)
    predictions = clf.predict(test_x)
    accuracy_score1 = accuracy_score(test_y, predictions)
    return accuracy_score1
def optimize_nearest_neighbours(train_x,train_y,test_x,test_y):
    best_score = 0
    best_n = 0
    for i in range(1,10):
        score = nearest_neighbours(train_x,train_y,test_x,test_y,n_neighbours=i)
        if score > best_score:
            best_score = score
            best_n = i
    return best_score, best_n
def predict_all_columns(train,test,function):
    columns = ["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    vectorizer = TfidfVectorizer()
    text = vectorizer.fit_transform(train["review_text"])
    text_test = vectorizer.transform(test["review_text"])
    for column in columns:
        result = function(text,train[column],text_test,test[column])
        print(column,":",result)
if __name__ == '__main__':
    #load the datasets and clean it
    print("runing dataAnalyse.py")
    test = pd.read_csv("data/test_clean.csv")
    train = pd.read_csv("data/train_clean.csv")
    validation = pd.read_csv("data/validation_clean.csv")
    predict_all_columns(train,test,optimize_nearest_neighbours)
