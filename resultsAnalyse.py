import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import shutil
from models import  runClassifierChain,columns,predict_multilabel_classifier,runClassifierChain
from sklearn.metrics import multilabel_confusion_matrix,classification_report,ConfusionMatrixDisplay, hamming_loss,confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance
from models import modelsDic,predict_multilabel_classifier,vectorize_data
from xgboost import XGBClassifier
def plot_orderedBy(results,metric,ascending=False):
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    print("Results for metric: ", metric)
    for column in columns:
        print("Results for column: ", column)
        # Plot the bar graph using seaborn
        k=10
        results['VARIABLE'] = results["MODEL"].apply(lambda x: x if x == "SVC" else x[:-10]) + "\n" + results["DATASET"].apply(lambda x: x.split("_")[1])
        top_k = results[results["TARGET"]==column].sort_values(by=metric,ascending=ascending).head(k)
        min_percentage = top_k[metric].min()
        max_percentage = top_k[metric].max()
        delta = max_percentage - min_percentage
        plt.figure(figsize=(17, 6))
        if(ascending == True):
            plt.ylim(0, max_percentage+delta)
        else:
            plt.ylim(min_percentage-delta, max_percentage+delta)
        sns.barplot(x='VARIABLE', y=metric, data=top_k, palette='viridis')
        plt.xlabel(metric)
        plt.ylabel('VARIABLE')
        plt.title(f'Top {k} {metric} for {column}')
        plt.show()

def plot_BestModel_orderedBy(metric,ascending=False):
    os.makedirs("plots/BestModel")
def mergeResults():
    df1 = pd.read_csv("ResultsArchive/results.csv")
    df1 = df1.rename(columns={'Sampling':'SAMPLING'})
    df2= pd.read_csv("ResultsArchive/results_all.csv")
    print("df1 columns:", df1.columns.tolist())
    print("df2 columns:", df2.columns.tolist())
    merged_df = pd.merge(df1, df2, how='outer', indicator=True)
    merged_df.to_csv("ResultsArchive/merged_results.csv", index=False)

def plot_BestModel_orderedBy(results,metric,ascending=False):
    os.makedirs("plots/BestModel",exist_ok=True)
    os.makedirs("plots/BestModel/"+metric)
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    print("Results for metric: ", metric)

    for column in columns:
        print("Results for column: ", column)
        # Plot the bar graph using seaborn
        k=10
        results['VARIABLE'] = results["MODEL"].apply(lambda x: x if x == "SVC" else x[:-10]) + "\n" + results["DATASET"].apply(lambda x: x.split("_")[1])
        top_k = results[results["TARGET"]==column].sort_values(by=metric,ascending=ascending).drop_duplicates("MODEL").head(k)
        min_percentage = top_k[metric].min()
        max_percentage = top_k[metric].max()
        delta = max_percentage - min_percentage
        plt.figure(figsize=(17, 6))
        if(ascending == True):
            plt.ylim(0, max_percentage+delta)
        else:
            plt.ylim(min_percentage-delta, max_percentage+delta)
        sns.barplot(x='VARIABLE', y=metric, data=top_k, palette='viridis')
        plt.xlabel(metric)
        plt.ylabel('VARIABLE')
        plt.title(f'Top {k} {metric} for {column}')

        plt.savefig(f"plots/BestModel/{metric}/{column}.png")

def best_models(metric,topK,ascending=False):
    results = pd.read_csv("ResultsArchive/results.csv")
    # Step 1: Get top k models per target
    top_models_per_target = (
        results.groupby("TARGET", group_keys=False)
        .apply(lambda group: group.sort_values(metric, ascending=ascending).drop_duplicates("MODEL").head(topK))
    )

    # Step 2: Get the set of models for each target
    unique_models_per_target = (
        top_models_per_target.groupby("TARGET")["MODEL"]
        .apply(set)
    )

    # Step 3: Find the intersection (models common to all targets)
    common_models = set.intersection(*unique_models_per_target)
    print("\n\nCommon models for all targets:", list(common_models))
def evaluateMultiLabelClassifier(model_name, dataset_type):
    train_df = pd.read_csv(f"data/{dataset_type}/train_clean.csv")
    val_df = pd.read_csv(f"data/{dataset_type}/validation_clean.csv")
    test_df = pd.read_csv(f"data/{dataset_type}/test_clean.csv")
    model = modelsDic[model_name]
    br = BinaryRelevance(classifier=model, require_dense=[False, True])
    train_x, val_x,test_x = vectorize_data(train_df, val_df, test_df)
    br.fit(train_x, train_df[columns])
    predictions = br.predict(test_x)
    os.makedirs(f"plots/ConfusionMatrix/{model_name}",exist_ok=True)
    drawConfusionMatrix(predictions,test_df[columns],model_name)
def drawConfusionMatrix(predictions, y_test,modelName):
    mcm = multilabel_confusion_matrix(predictions, y_test)
    create_single_confusion_matrix(mcm,modelName)
    # Visualize each matrix
    
    for i, (matrix, label) in enumerate(zip(mcm, columns)):
        # Plot the confusion matrix for each label
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted No', 'Predicted Yes'],
                    yticklabels=['Actual No', 'Actual Yes'])
        plt.title(f'Confusion Matrix for {label}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"plots/ConfusionMatrix/{modelName}/{label}.png")
def create_single_confusion_matrix(mcm,name):
    micro_cm = np.sum(mcm, axis=0)  # Sum all individual confusion matrices

    # Plot the micro-averaged confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(micro_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Micro-Averaged Confusion Matrix\n(Sum of all label confusion matrices)')
    plt.colorbar()

    # Add text annotations
    thresh = micro_cm.max() / 2.
    for i in range(micro_cm.shape[0]):
        for j in range(micro_cm.shape[1]):
            plt.text(j, i, format(micro_cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if micro_cm[i, j] > thresh else "black")

    # Label the axes
    class_names = ['Negative', 'Positive']
    plt.xticks([0, 1], [f'Predicted {name}' for name in class_names])
    plt.yticks([0, 1], [f'Actual {name}' for name in class_names])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig(f"plots/ConfusionMatrix/{name}/micro_averaged.png")

def best_performing_datasets(metric,topK,ascending=False):
    results = pd.read_csv("ResultsArchive/results.csv")
    # Step 1: Get top k models per target
    top_models_per_target = (
        results.groupby("TARGET", group_keys=False)
        .apply(lambda group: group.sort_values(metric, ascending=ascending).drop_duplicates("DATASET").head(topK))
    )

    # Step 2: Get the set of models for each target
    unique_models_per_target = (
        top_models_per_target.groupby("TARGET")["DATASET"]
        .apply(set)
    )

    # Step 3: Find the intersection (models common to all targets)
    #common datasets in the topK for each target
    common_datasets = set.intersection(*unique_models_per_target)
    print("\n\nCommon datasets for all targets:", list(common_datasets))

def getBestModelByTarget(metric,ascending=False):
    results = pd.read_csv("ResultsArchive/results.csv")
    return1 = []
    for target in columns:
        top_models = results[(results["TARGET"]==target) & (results["MODEL"] != "VotingClassifier")].sort_values(by=metric,ascending=ascending).drop_duplicates("MODEL").head(1)
        #print(f"Best model for {target} is {top_models['MODEL'].values[0]} with {metric} = {top_models[metric].values[0]}")
        return1.append(top_models['MODEL'].values[0])
    return return1
    
def get_bar_graph_average_performance_of_datasets(df):
    df = df.groupby("SAMPLING")[["ACCURACY","F1","RECALL","PRECISION","HAMMING_LOSS"]].mean().reset_index()

    # remove everyletter until you find _ on the dataset column
    df['SAMPLING'] = df['SAMPLING'].str.replace(r'^[^_]*_', '', regex=True)


    # Plot the bar graph using seaborn
    plt.figure(figsize=(17, 6))
    sns.barplot(x='SAMPLING', y='HAMMING_LOSS', data=df, palette='viridis')
    plt.xlabel('Dataset Treatment')
    plt.ylabel('Hamming Loss Score')
    plt.title('Average Performance of Datasets')

    # Set y-axis limits (lower limit set to 0.08)
    plt.ylim(df['HAMMING_LOSS'].min()- 0.0005, df['HAMMING_LOSS'].max() + 0.0005)  # Add a small margin to the upper limit
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.savefig(f"plots/bar_graphs/dataset_performance.png")
def get_bar_graph_average_performance_of_model(metric,df,ascending=False):
    df = df.groupby("MODEL")[["ACCURACY","F1","RECALL","PRECISION","HAMMING_LOSS"]].mean().reset_index()
    df.sort_values(by=metric, ascending=ascending, inplace=True)
    # Plot the bar graph using seaborn
    plt.figure(figsize=(17, 6))
    sns.barplot(x='MODEL', y=metric, data=df, palette='viridis')
    plt.xlabel('model')
    plt.ylabel(metric+' Score')
    plt.title('Average Performance of Models')

    # Set y-axis limits (lower limit set to 0.08)
    plt.ylim(df[metric].min()- 0.01, df[metric].max() + 0.01)  # Add a small margin to the upper limit«

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.savefig(f"plots/bar_graphs/model_performance.png")

def get_bar_graph_of_top_k_models_by_metric(k,metric,df,ascending=False):
    # Get the top 5 models for each dataset
    top_5_models = df.sort_values(by=metric, ascending=ascending).drop_duplicates("MODEL").head(k)
    top_5_models["graph_label"] = top_5_models["MODEL"]
    # Plotting
    plt.figure()
    sns.barplot(x='graph_label', y=metric, data=top_5_models, palette='viridis')
    plt.xlabel('Model')
    plt.ylabel(metric+' Score')
    plt.title('Top 5 Models')
    plt.ylim(top_5_models[metric].min()- top_5_models[metric].min()/4, top_5_models[metric].max() + 0.001)
        
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"plots/bar_graphs/top_5_models_per_dataset.png")

def get_failed_queries_of_model(model, train_df, val_df, test_df):
    columns = ["ENTREGA", "OUTROS", "PRODUTO", "CONDICOESDERECEBIMENTO", "ANUNCIO"]

    # remove every row with the col INADEQUADA 1
    train_df = train_df[train_df["INADEQUADA"] != 1]
    val_df = val_df[val_df["INADEQUADA"] != 1]
    test_df = test_df[test_df["INADEQUADA"] != 1]
    # remove the column INADEQUADA
    train_df = train_df.drop(columns=["INADEQUADA"])
    val_df = val_df.drop(columns=["INADEQUADA"])
    test_df = test_df.drop(columns=["INADEQUADA"])
    # Vectorize the text data (convert to numeric format)
    train_X, val_X, test_X = vectorize_data(train_df, val_df, test_df)

    # Ensure labels are numeric
    train_y = train_df[columns]
    val_y = val_df[columns]
    test_y = test_df[columns]

    # Create a Binary Relevance classifier with the specified model
    br = BinaryRelevance(classifier=model, require_dense=[False, True])

    # Fit the model on the training data
    br.fit(train_X, train_df[columns])

    # Predict on the validation and test data
    val_predictions = br.predict(val_X)
    test_predictions = br.predict(test_X)

    # Convert predictions to dense format for comparison
    val_predictions = val_predictions.toarray()
    test_predictions = test_predictions.toarray()

    # Print predictions
    hamming_loss1 = hamming_loss(test_y, test_predictions)
    print("Hamming Loss:", hamming_loss1)

    # Identify mislabeled reviews in the validation set
    val_mislabeled_indices = (val_predictions != val_y.values).any(axis=1)
    val_mislabeled_reviews = val_df[val_mislabeled_indices].copy()
    val_mislabeled_reviews["True Labels"] = val_y[val_mislabeled_indices].values.tolist()
    val_mislabeled_reviews["Predicted Labels"] = val_predictions[val_mislabeled_indices].tolist()

    # Identify mislabeled reviews in the test set
    test_mislabeled_indices = (test_predictions != test_y.values).any(axis=1)
    test_mislabeled_reviews = test_df[test_mislabeled_indices].copy()
    test_mislabeled_reviews["True Labels"] = test_y[test_mislabeled_indices].values.tolist()
    test_mislabeled_reviews["Predicted Labels"] = test_predictions[test_mislabeled_indices].tolist()

    # Combine validation and test mislabeled reviews into a single DataFrame
    mislabeled_reviews = pd.concat([val_mislabeled_reviews, test_mislabeled_reviews], ignore_index=True)

    # Save the mislabeled reviews to a CSV file
    mislabeled_reviews.to_csv("ResultsArchive/mislabeled_reviews_with_predictions.csv", index=False)

if __name__ == '__main__':
    #best_models("ACCURACY",5,ascending=False)
    #best_performing_datasets("ACCURACY",5,ascending=False)
    # for the top 10 performing models all had only the datasets_stem_text in the TOP 5
    # Oversampling was the best performing sampling technique
    # the best performing 'MLPClassifier', 'VotingClassifier'

    results = pd.read_csv("ResultsArchive/results_all.csv")
    #get_bar_graph_average_performance_of_datasets(results)
    #get_bar_graph_average_performance_of_model("F1",results)
    
    #get_bar_graph_of_top_k_models_by_metric(10,"F1",results)
    evaluateMultiLabelClassifier("KNeighborsClassifier","datasets_stem_text")

    """
    if os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots")
    os.makedirs("plots/ConfusionMatrix",exist_ok=True)
    #mergeResults()
    #models = getBestModelByTarget("F1",ascending=False)
    #runClassifierChain("OVERSAMPLING","datasets_stem_text",models)
    evaluateMultiLabelClassifier("KNeighborsClassifier","datasets_stem_text")
    evaluateMultiLabelClassifier("VotingClassifier","datasets_stem_text")
    print("ploting ordered by")
    results = pd.read_csv("ResultsArchive/results.csv")# change this to see the statistics for the the multilabel models
    plot_BestModel_orderedBy(results,"ACCURACY",ascending=False)
    plot_BestModel_orderedBy(results,"F1",ascending=False)
    plot_BestModel_orderedBy(results,"RECALL",ascending=False)
    plot_BestModel_orderedBy(results,"PRECISION",ascending=False)
    plot_BestModel_orderedBy(results,"HAMMING_LOSS",ascending=True)
    """
    """
    plot_orderedBy(results,"ACCURACY",ascending=False)
    plot_orderedBy(results,"F1",ascending=False)
    plot_orderedBy(results,"RECALL",ascending=False)
    plot_orderedBy(results,"PRECISION",ascending=False)
    plot_orderedBy(results,"HAMMING_LOSS",ascending=True)"
    """
    dataset_without_num_train = pd.read_csv("data/datasets_removeNumbers/train_clean.csv")
    dataset_without_num_val = pd.read_csv("data/datasets_removeNumbers/validation_clean.csv")
    dataset_without_num_test = pd.read_csv("data/datasets_removeNumbers/test_clean.csv")
    get_failed_queries_of_model(XGBClassifier(),dataset_without_num_train,dataset_without_num_val,dataset_without_num_test)
