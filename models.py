import numpy as np

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
# extra models check if possible
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time


import pandas as pd

# dataset balancing

def undersampling(train_x,train_y):
    rus = RandomUnderSampler(random_state=42)
    train_x_os, train_y_os = rus.fit_resample(train_x, train_y)
    return (train_x_os, train_y_os)

def oversampling(train_x,train_y):
    ros = RandomOverSampler(random_state=42)
    train_x_us, train_y_us = ros.fit_resample(train_x, train_y)
    return (train_x_us,train_y_us)

def generate_balanced_data(text_train,train, columns):
    datasets = {}
    for column in columns:
        datasets[column] = {"UNDERSAMPLING":undersampling(text_train,train[column]), "OVERSAMPLING":oversampling(text_train,train[column]), "ORIGINAL":(text_train,train[column])}
    return datasets

# vectorize the data
def vectorize_data(train, test):
    vectorizer = TfidfVectorizer()
    text_train = vectorizer.fit_transform(train["review_text"])
    text_test = vectorizer.transform(test["review_text"])
    return text_train, text_test

#nearest neighbours optimization
        
def nearest_neighbours(train_x,train_y,test_x,test_y, n_neighbours=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbours)
    clf.fit(train_x,train_y)
    predictions = clf.predict(test_x)
    accuracy_score1 = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
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

#run a lot of models
def run_extra_models(datasets,results,columns):
    models = [
    ('XGBoost', XGBClassifier()),
    ('AdaBoost', AdaBoostClassifier(n_estimators=50,algorithm='SAMME')),
    ('Bagging', BaggingClassifier(n_estimators=50, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50)),
    #('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    #('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
    ('Mlp-adam', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='adam', learning_rate='constant')),
    ('Mlp-lbfgs', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='lbfgs', learning_rate='constant')),
    ('Mlp-sgd', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='sgd', learning_rate='constant')),
    ('CatBoost', CatBoostClassifier(n_estimators=50,logging_level='Silent'))
]   
    kfold = KFold(n_splits=3, random_state=1, shuffle=True)
    for column in columns:
        dataset = datasets[column]
        for dataset_type,dataset in dataset.items():
            text,target = dataset
            print("Running models for column: ", column)
            for name, model in models:
                cv_results = cross_val_score(model, text, target, cv=kfold, scoring='accuracy')
                print(f"{name}: {cv_results} accuracy")
                entry = {"MODEL":model.__class__.__name__,"DATASET":dataset_type,"ACCURACY":cv_results.mean(),'TARGET':column}
                results = addToDf(results,entry)
    return results
#predict using multilabel classifier
def predict_multilabel_classifier(train_x,train_y,test_x,test_y,results,model,model_name):
    br = BinaryRelevance(classifier=model, require_dense=[False, True])
    br.fit(train_x, train_y)
    predictions = br.predict(test_x)
    accuracy_score1 = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average='micro')
    recall = recall_score(test_y, predictions, average='micro')
    precision = precision_score(test_y, predictions, average='micro')
    entry = {"MODEL":model_name,"DATASET":"ORIGINAL","ACCURACY":accuracy_score1,"F1":f1,"RECALL":recall,"PRECISION":precision,'TARGET':"All"}
    results = addToDf(results,entry)
    return results
def multilabel_a_lot_of_models(train_x,train_y,test_x,test_y,results):
    models = [
        ('KNN', KNeighborsClassifier()),
        #('Radius Neighbors', RadiusNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('SVM', SVC()),
        ('XGBoost', XGBClassifier()),
        ('AdaBoost', AdaBoostClassifier(algorithm='SAMME')),
        ('Bagging', BaggingClassifier( n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50)),
        #('GaussianNB', GaussianNB()),
        ('random Forest',RandomForestClassifier()),
        #('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
        #('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
        ('Mlp-adam', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='adam', learning_rate='constant')),
        ('Mlp-lbfgs', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='lbfgs', learning_rate='constant')),
        ('Mlp-sgd', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='sgd', learning_rate='constant')),
        ('CatBoost', CatBoostClassifier(n_estimators=50,logging_level='Silent'))
    ]
    voting = VotingClassifier(estimators=models, voting='soft')
    #staking = StackingClassifier(estimators=models)
    models.append(('Voting',voting))
    #models.append(('Stacking',staking))
    for model_name,model in models:
        print("Running model: ", model_name)
        time1 = time.time()

        results = predict_multilabel_classifier(train_x,train_y,test_x,test_y,results,model,model_name)
        time2 = time.time()
        print("Time: ", time2-time1)
    return results



#single labeling for each column in singular
def predict_all_columns(datasets,text_test,function,results,columns):

    for column in columns:
        print("Predicting column: ", column)
        dataset = datasets[column]
        for dataset_type,dataset in dataset.items():
            text,target = dataset
            result = function(text,target,text_test,test[column])
            f1 = result[2][1]
            precision = result[2][3]
            recall = result[2][2]
            accuracy = result[2][0]
            entry = {"MODEL":function.__name__,"DATASET":dataset_type,"PRECISION":precision, "ACCURACY":accuracy,"F1":f1,"RECALL":recall,"MODEL_PARAMS":result[1],'TARGET':column}
            results = addToDf(results,entry)
    return results
def addToDf(results,entry):
    if results.empty:
        results = pd.DataFrame([entry])

    else:
        results = pd.concat([results, pd.DataFrame([entry]).reindex(columns=results.columns)], ignore_index=True)
    return results
def generateDf(columns):
        results = pd.DataFrame([],  columns =  ["MODEL","DATASET","PRECISION", "ACCURACY","F1","RECALL","MODEL_PARAMS","TARGET"])
        results["TARGET"] = pd.Categorical([], categories=columns)
        results["DATASET"] = pd.Categorical([], categories=["UNDERSAMPLING", "OVERSAMPLING", "ORIGINAL"])
        return results
if __name__ == '__main__':
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    results = generateDf(columns)
    #load the datasets and clean it
    print("runing dataAnalyse.py")
    test = pd.read_csv("data/test_clean.csv")
    train = pd.read_csv("data/train_clean.csv")
    validation = pd.read_csv("data/validation_clean.csv")
    #vectorize the review strings
    text_train, text_test = vectorize_data(train, test)
    datasets = generate_balanced_data(text_train,train,columns)
    #results = predict_all_columns(datasets,text_test,optimize_nearest_neighbours,results,columns)
    #results_extra =run_extra_models(datasets,results,columns)
    #results_extra.to_csv("./results/results_extra.csv")
    results = multilabel_a_lot_of_models(text_train,train[columns],text_test,test[columns],results)
    results.to_csv("./results/results.csv")
