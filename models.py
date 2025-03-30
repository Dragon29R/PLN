import numpy as np
import os
import copy
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,hamming_loss,recall_score,f1_score,precision_score,roc_curve, RocCurveDisplay,jaccard_score
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
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time


import pandas as pd

#Globals
#target variables
columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
# models to try
models = [
    ('KNN', KNeighborsClassifier()),
    #('Radius Neighbors', RadiusNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC(probability=True)),
    ('XGBoost', XGBClassifier()),
    ('AdaBoost', AdaBoostClassifier(algorithm='SAMME')),
    ('Bagging', BaggingClassifier( n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50)),
    #('GaussianNB', GaussianNB()),
    ('random Forest',RandomForestClassifier()),
    #('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    #('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
    ('Mlp-adam', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='adam', learning_rate='constant')),
    ('Mlp-lbfgs', MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500,early_stopping=True, n_iter_no_change=5, solver='lbfgs', learning_rate='constant')),
    ('Mlp-sgd', MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, n_iter_no_change=5, solver='sgd', learning_rate='constant')),
    ('CatBoost', CatBoostClassifier(n_estimators=50,logging_level='Silent'))
    ]
# add the voting classifier
voting_estimators = [(name, model) for name, model in models]
voting = VotingClassifier(estimators=voting_estimators, voting='soft')
#staking = StackingClassifier(estimators=models)
models.append(('Voting',voting))

#dictionary of models
# create a dictionary of models
modelsDic = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SVC': SVC(probability=True),
    'XGBClassifier': XGBClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(algorithm='SAMME'),
    'BaggingClassifier': BaggingClassifier( n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=50),
    #('GaussianNB', GaussianNB()),
    'RandomForestClassifier':RandomForestClassifier(),
    #('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    #('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500,early_stopping=True, n_iter_no_change=5, solver='lbfgs', learning_rate='constant'),
    'CatBoostClassifier': CatBoostClassifier(n_estimators=50,logging_level='Silent'),
    'VotingClassifier': VotingClassifier(estimators=[(name, model) for name, model in models], voting='soft')
}


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
def vectorize_data(train,validate, test):
    vectorizer = TfidfVectorizer()
    text_train = vectorizer.fit_transform(train["review_text"])
    text_validate = vectorizer.transform(validate["review_text"])
    text_test = vectorizer.transform(test["review_text"])
    return text_train, text_validate,text_test

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
#create nearest neighbours model
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


#run all the models for each column to try to find which one works best for each column using each dataset
    """
    Runs all the models for each column to try to find which one works best for each column using each dataset.

    Parameters
    ----------
    datasets : dict
        A dictionary where the key is the column name and the value is a dictionary with the different sampling methods.
    validation_x : array
        The validation data.
    validation_y : array
        The validation labels.
    results : DataFrame
        The DataFrame where the results will be stored.
    columns : list
        The list of columns to consider.
    dataset_type : str
        The type of dataset (e.g. "stem_text").

    Returns
    -------
    results : DataFrame
        The DataFrame with the results.
    """
def run_extra_models(datasets,validation_x,validation_y,results,columns,dataset_type):
    for column in columns:
        dataset = datasets[column]
        for sampling,dataset in dataset.items():
            train_x,train_y = dataset
            print("Running models for column: ", column)
            time1 = time.time()
            for name, model in models:
                #print("Running model: ", name)
                model.fit(train_x,train_y)
                predictions = model.predict(validation_x)
                accuracy_score1 = accuracy_score(validation_y[column], predictions)
                f1 = f1_score(validation_y[column], predictions, average='micro')
                recall = recall_score(validation_y[column], predictions, average='micro')
                precision = precision_score(validation_y[column], predictions, average='micro')
                hamming_loss1 = hamming_loss(validation_y[column], predictions)
                entry = {"MODEL":name,"DATASET":dataset_type,"Sampling":sampling,"ACCURACY":accuracy_score1,"F1":f1,"RECALL":recall,"PRECISION":precision,"HAMMING_LOSS":hamming_loss1,'TARGET':column}
                results = addToDf(results,entry)
            time2 = time.time()
            delta = time2-time1
            print("Time: ", time.strftime("%H:%M:%S", time.gmtime(delta)))
    return results
#predict using Binaryrelevance to predict multilabel using only one model for all the targets
def predict_multilabel_classifier(train_x,train_y,test_x,test_y,results,model,model_name,dataset_name,sampling):
    br = BinaryRelevance(classifier=model, require_dense=[False, True])
    br.fit(train_x, train_y)
    predictions = br.predict(test_x)
    accuracy_score1 = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average='micro')
    recall = recall_score(test_y, predictions, average='micro')
    precision = precision_score(test_y, predictions, average='micro')
    hamming_loss1 = hamming_loss(test_y, predictions)
    entry = {"MODEL":model_name,"DATASET":dataset_name,"SAMPLING":sampling,"ACCURACY":accuracy_score1,"F1":f1,"RECALL":recall,"PRECISION":precision,"HAMMING_LOSS":hamming_loss1,'TARGET':"All"}
    print("ResultsMM: ", results)
    results = addToDf(results,entry)
    return results
#predict using multilabel classifier for all the models for all the targets, it is faster than the previous function because it uses the same model for all columns 
def multilabel_a_lot_of_models(train_x,train_y,validation_x,validation_y,results,dataset_name):
    #models.append(('Stacking',staking))
    for model_name,model in models:
        print("Running model: ", model_name)
        time1 = time.time()
        results = predict_multilabel_classifier(train_x,train_y,validation_x,validation_y,results,model,model_name,"ORIGINAL",dataset_name)
        time2 = time.time()
        delta = time2-time1
        print("Time: ", time.strftime("%H:%M:%S", time.gmtime(delta)))
        print("Results multilabel: ", results)
    return results
# run classivier chain on the list of the models that 
def runClassifierChain(sampling,datasetType,modelsList):
    if os.path.exists("results/results_classifierChain.csv"):
        results = pd.read_csv("results/results_classifierChain.csv")
    else:
        results =generateDf(columns)
    test = pd.read_csv("data/"+datasetType+"/test_clean.csv")
    train = pd.read_csv("data/"+datasetType+"/train_clean.csv")
    validation = pd.read_csv("data/"+datasetType+"/validation_clean.csv")
    text_train,text_validate, text_test = vectorize_data(train, validation,test)
    datasets = generate_balanced_data(text_train,train,columns)
    datasets = {column:datasets[column][sampling] for column in columns}

#Get Best Models

    # Create and fit separate chains
    print("Running model: ", modelsList)
    # train the models individually
    listModels = []
    for i,modelname in enumerate(modelsList):
        model =copy.deepcopy( modelsDic[modelname])
        column = columns[i]
        train_x,train_y = datasets[column]
        model.fit(train_x, train_y)  # Fit on single label
        listModels.append(model)
        print("Training model: ", model.__class__.__name__)

    chains = []
    for i,modelname in enumerate(modelsList):
        column = columns[i]
        model =listModels[i]
        print("Running model: ", model.__class__.__name__, " for column: ", column)
        train_x,train_y = text_train , train[columns]
        chain = ClassifierChain(model, order='random', random_state=42)
        chain.fit(train_x, train_y)  # Fit on single label
        chains.append(chain)


    # Make predictions
    predictions = np.array([chain.predict(text_test) for chain in
                          chains])
    Y_pred_ensemble = predictions.mean(axis=0)
    ensemble_jaccard_score = jaccard_score(
    test[columns], Y_pred_ensemble >= 0.5, average="samples"
)
    print("Ensemble Jaccard score", ensemble_jaccard_score)
    average = precision_score(test[columns], Y_pred_ensemble >= 0.5, average="samples")
    f1_score1 = f1_score(test[columns], Y_pred_ensemble >= 0.5, average="samples")
    recall = recall_score(test[columns], Y_pred_ensemble >= 0.5, average="samples")
    accuracy = accuracy_score(test[columns], Y_pred_ensemble >= 0.5)
    entry = {"MODEL":modelsList,"DATASET":datasetType,"PRECISION":average, "ACCURACY":accuracy,"F1":f1_score1,"RECALL":recall,"MODEL_PARAMS":'','TARGET':"All"}
    results = addToDf(results,entry)
    results.to_csv("results/results_classifierChain.csv")
    print("Results: ", entry)
    # Evaluate
    """chain_jaccard_scores = [
        jaccard_score(test[columns], Y_pred_chain >= 0.5, average="samples")
        for Y_pred_chain in predictions
    ]"""


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
#add the entry to the dataframe
def addToDf(results,entry):
    if results.empty:
        results = pd.DataFrame([entry])

    else:
        results = pd.concat([results, pd.DataFrame([entry]).reindex(columns=results.columns)], ignore_index=True)
    return results
#generate the dataframe with the columns
def generateDf(columns):
        results = pd.DataFrame([],  columns =  ["MODEL","DATASET","PRECISION", "ACCURACY","F1","RECALL","MODEL_PARAMS","TARGET"])
        results["TARGET"] = pd.Categorical([], categories=columns)
        results["Sampling"] = pd.Categorical([], categories=["UNDERSAMPLING", "OVERSAMPLING", "ORIGINAL"])
        return results

if __name__ == '__main__':
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    datasetsTypes = ["datasets_all","datasets_lemmatize_text","datasets_normalizeRepeatedChars",
                     "datasets_removeNulls","datasets_removeNumbers","datasets_removePonctuation","datasets_removeStopwords",
                     "datasets_removeUpper","datasets_stem_text","datasets_tokenize_text"]
    results = generateDf(columns)
    #load the datasets and clean it
    print("runing dataAnalyse.py")
    for datasetType in datasetsTypes:
        print("Running dataset: ", datasetType)
        time1 = time.time()
        test = pd.read_csv("data/"+datasetType+"/test_clean.csv")
        train = pd.read_csv("data/"+datasetType+"/train_clean.csv")
        validation = pd.read_csv("data/"+datasetType+"/validation_clean.csv")
        #vectorize the review strings
        text_train,text_validate, text_test = vectorize_data(train, validation,test)
        datasets = generate_balanced_data(text_train,train,columns)
        #results = predict_all_columns(datasets,text_test,optimize_nearest_neighbours,results,columns)
        #results =run_extra_models(datasets,text_validate,validation[columns],results,columns,datasetType)
        results = multilabel_a_lot_of_models(text_train,train[columns],text_validate,validation[columns],results,datasetType)
        time2 = time.time()
        delta = time2-time1
        print(datasetType+" TimeElapsed: ", time.strftime("%H:%M:%S", time.gmtime(delta)))
        results.to_csv("./results/results_all.csv")
    #results_extra.to_csv("./results/results_extra.csv")
    print("Results: ", results)
    if not os.path.exists("results"):
        os.makedirs("results")
    results.to_csv("./results/results.csv")