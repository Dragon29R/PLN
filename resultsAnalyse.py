import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from models import  runClassifierChain,columns,predict_multilabel_classifier,runClassifierChain

def plot_orderedBy(metric,ascending=False):
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    print("Results for metric: ", metric)
    results = pd.read_csv("ResultsArchive/results.csv")
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
    os.makedirs("plots/BestModel/"+metric)
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    print("Results for metric: ", metric)
    results = pd.read_csv("ResultsArchive/results.csv")
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
    df = df.groupby("DATASET")[["ACCURACY","F1","RECALL","PRECISION","HAMMING_LOSS"]].mean().reset_index()

    # remove everyletter until you find _ on the dataset column
    df['DATASET'] = df['DATASET'].str.replace(r'^[^_]*_', '', regex=True)


    # Plot the bar graph using seaborn
    plt.figure(figsize=(17, 6))
    sns.barplot(x='DATASET', y='HAMMING_LOSS', data=df, palette='viridis')
    plt.xlabel('Dataset Treatment')
    plt.ylabel('Hamming Score')
    plt.title('Average Performance of Datasets')

    # Set y-axis limits (lower limit set to 0.08)
    plt.ylim(df['HAMMING_LOSS'].min()- 0.01, df['HAMMING_LOSS'].max() + 0.01)  # Add a small margin to the upper limit

    plt.tight_layout()  # Ensure everything fits within the figure
    plt.savefig(f"plots/bar_graphs/dataset_performance.png")

def get_bar_graph_average_performance_of_model(df):
    df = df.groupby("MODEL")[["ACCURACY","F1","RECALL","PRECISION","HAMMING_LOSS"]].mean().reset_index()



    # Plot the bar graph using seaborn
    plt.figure(figsize=(17, 6))
    sns.barplot(x='MODEL', y='HAMMING_LOSS', data=df, palette='viridis')
    plt.xlabel('model')
    plt.ylabel('Hamming Score')
    plt.title('Average Performance of Models')

    # Set y-axis limits (lower limit set to 0.08)
    plt.ylim(df['HAMMING_LOSS'].min()- 0.01, df['HAMMING_LOSS'].max() + 0.01)  # Add a small margin to the upper limit

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.savefig(f"plots/bar_graphs/model_performance.png")


if __name__ == '__main__':
    #best_models("ACCURACY",5,ascending=False)
    #best_performing_datasets("ACCURACY",5,ascending=False)
    # for the top 10 performing models all had only the datasets_stem_text in the TOP 5
    # Oversampling was the best performing sampling technique
    # the best performing 'MLPClassifier', 'VotingClassifier'

    results = pd.read_csv("ResultsArchive/results.csv")
    get_bar_graph_average_performance_of_datasets(results)
    get_bar_graph_average_performance_of_model(results)

    """ is this?
    if os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots")
    models = getBestModelByTarget("F1",ascending=False)
    runClassifierChain("OVERSAMPLING","datasets_stem_text",models)
    """


    """plot_BestModel_orderedBy("ACCURACY",ascending=False)
    plot_orderedBy("ACCURACY",ascending=False)
    plot_orderedBy("F1",ascending=False)
    plot_orderedBy("RECALL",ascending=False)
    plot_orderedBy("PRECISION",ascending=False)
    plot_orderedBy("HAMMING_LOSS",ascending=True)"""
