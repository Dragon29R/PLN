import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


import pandas as pd
def undersampling(train_x,train_y):
    rus = RandomUnderSampler(random_state=42)
    train_x_os, train_y_os = rus.fit_resample(train_x, train_y)
    return train_x_os, train_y_os
def oversampling(train_x,train_y):
    ros = RandomOverSampler(random_state=42)
    train_x_us, train_y_us = ros.fit_resample(train_x, train_y)
    return train_x_us,train_y_us

def nearest_neighbours(train_x,train_y,test_x,test_y, n_neighbours=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbours)
    clf.fit(train_x,train_y)
    predictions = clf.predict(test_x)
    accuracy_score1 = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average='weighted')
    recall = recall_score(test_y, predictions, average='weighted')
    precision = precision_score(test_y, predictions, average='weighted')
    return [accuracy_score1,f1,recall,precision]
def optimize_nearest_neighbours(train_x,train_y,test_x,test_y):
    best_score = 0
    best_stats =[]
    best_n = 0
    for i in range(1,15):
        stats = nearest_neighbours(train_x,train_y,test_x,test_y,n_neighbours=i)
        score = stats[0]
        if score > best_score:
            best_score = score
            best_n = i
            best_stats = stats
    return best_score, best_n , best_stats

def predict_all_columns(train,test,function,results):
    columns = ["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    vectorizer = TfidfVectorizer()
    text = vectorizer.fit_transform(train["review_text"])
    text_test = vectorizer.transform(test["review_text"])
    for column in columns:
        result = function(text,train[column],text_test,test[column])
        f1 = result[2][1]
        precision = result[2][3]
        recall = result[2][2]
        accuracy = result[2][0]
        results = results.append({"MODEL":"KNN","PRECISION":precision, "ACCURACY":accuracy,"F1":f1,"RECALL":recall,"MODEL_PARAMS":result[1],'TARGET':column}, ignore_index=True)
        print(column,":",result)
if __name__ == '__main__':
    results = pd.DataFrame([],  columns =  ["MODEL","PRECISION", "ACCURACY","F1","RECALL","MODEL_PARAMS","TARGET"])
    results["TARGET"] = pd.Categorical([], categories=["ENTREGA", "OUTROS", "PRODUTO", "CONDICOESDERECEBIMENTO", "ANUNCIO"])
    #load the datasets and clean it
    print("runing dataAnalyse.py")
    test = pd.read_csv("data/test_clean.csv")
    train = pd.read_csv("data/train_clean.csv")
    validation = pd.read_csv("data/validation_clean.csv")
    results = predict_all_columns(train,test,optimize_nearest_neighbours,results)
