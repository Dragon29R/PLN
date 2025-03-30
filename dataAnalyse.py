import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from dataprocessing import generate_corpus,generate_corpus_no_stemma
import os



# Create directories if they do not exist
def create_directories():
    os.makedirs("plots/boxplots", exist_ok=True)
    os.makedirs("plots/wordclouds", exist_ok=True)
    os.makedirs("plots/bar_graphs", exist_ok=True)

# Plot boxplot of the review_text length
def plot_boxplot(data,legend):
    sizes = [len(i) for i in data["review_text"]]
    max1 = data[data["review_text"].str.len()==max(sizes)]
    plt.figure()
    plt.boxplot(sizes)
    plt.title(legend)
    plt.ylabel("Review length")
    # Create legend without handles
    legend_texts = [
    "Max review length: " + str(max1["review_text"].str.len().values[0]),
    "Average review length: " + str(sum(sizes)/len(sizes))
]
    plt.legend(legend_texts)

    plt.savefig("plots/boxplots/"+legend+"_boxplot.png")

def create_a_bar_graph(data,legend):
    #create a graph bar in wich we see the number of reviews per tag
    tags = ["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    class_counts = {tag: data[tag].sum() for tag in tags}  # Sum the values for each class
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title(f"Number of Reviews per Class - {legend}")
    plt.xlabel("Class")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/bar_graphs/{legend}_class_distribution.png")



# Generate the wordcloud
def generateWordCloud(corpus,title):
    wordcloud = WordCloud().generate(" ".join(corpus))
    plt.figure()
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.savefig("plots/wordclouds/wordcloud_" + title + ".png")

# Plot wordclouds
def plot_wordclouds(df):
    sw = set(nltk.corpus.stopwords.words('portuguese'))
    ps = SnowballStemmer('portuguese')
    tags = ["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    corpus = generate_corpus_no_stemma(df,sw)
    generateWordCloud(corpus,"all")
    for tag in tags:
        df1 = df[df[tag]==1]
        corpus = generate_corpus_no_stemma(df1,sw)
        generateWordCloud(corpus,tag)

if __name__ == '__main__':
    create_directories()
    #load the datasets and clean it
    print("runing dataAnalyse.py")
    test_clean = pd.read_csv("data/test.csv")
    train_clean = pd.read_csv("data/train.csv")
    validation_clean = pd.read_csv("data/validation.csv")

    #plot some analysis
    plot_boxplot(train_clean,"train dataset")
    plot_boxplot(test_clean,"test dataset")
    plot_boxplot(validation_clean,"validation dataset")

    #join everything in one dataset
    entire_dataset = pd.concat([train_clean,validation_clean,test_clean])

    plot_boxplot(entire_dataset,"dataset")
    create_a_bar_graph(entire_dataset,"dataset")

    #print the review with the max length
    max1 = entire_dataset[entire_dataset["review_text"].str.len()==max([len(i) for i in entire_dataset["review_text"]])]
    print("Max review length: ",max1["review_text"].str.len().values[0],"Review: ",max1["review_text"].values[0])

    #draw wordclouds
    plot_wordclouds(entire_dataset)