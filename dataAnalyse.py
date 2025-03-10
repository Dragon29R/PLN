import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from dataprocessing import generate_corpus


# Plot boxplot of the review_text length
def plot_boxplot(data,legend):
    sizes = [len(i) for i in data["review_text"]]
    max1 = data[data["review_text"].str.len()==max(sizes)]
    plt.boxplot(sizes)
    plt.title(legend)
    plt.savefig("plots/boxplots/"+legend+"_boxplot.png")

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
    corpus = generate_corpus(df,sw,ps)
    generateWordCloud(corpus,"all")
    for tag in tags:
        df1 = df[df[tag]==1]
        corpus = generate_corpus(df1,sw,ps)
        generateWordCloud(corpus,tag)

if __name__ == '__main__':
    #load the datasets and clean it
    print("runing dataAnalyse.py")
    test_clean = pd.read_csv("data/test_clean.csv")
    train_clean = pd.read_csv("data/train_clean.csv")
    validation_clean = pd.read_csv("data/validation_clean.csv")

    #plot some analysis
    plot_boxplot(train_clean,"train")
    plot_boxplot(test_clean,"test")
    plot_boxplot(validation_clean,"validation")
    #draw wordclouds
    plot_wordclouds(train_clean)