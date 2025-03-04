import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud
import nltk
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import re
def remove_noise(df,name):
    df1 = df[df["review_text"].notnull()]
    df2 = df1[df1["review_text"]!=""]
    df3 = df2.dropna()
    df_clean = df3[df3["INADEQUADA"]==0]
    print(str(name)+":",len(df),str(name)+"_clean:",len(df_clean))
    return df_clean
def plot_boxplot(data,legend):
    sizes = [len(i) for i in data["review_text"]]
    max1 = data[data["review_text"].str.len()==max(sizes)]
    plt.boxplot(sizes)
    plt.title(legend)
    plt.savefig("plots/boxplots/"+legend+"_boxplot.png")

def remove_stopwords(data):
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    data["review_text"] = data["review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
# remove stop words and apply stemming
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
def generateWordCloud(corpus,title):
    wordcloud = WordCloud().generate(" ".join(corpus))
    plt.figure()
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.savefig("plots/wordclouds/wordcloud_" + title + ".png")

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
    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")

    test_clean = remove_noise(test,"test")
    train_clean = remove_noise(train,"train")
    validation_clean = remove_noise(validation,"validation")
    #plot some analysis
    plot_boxplot(train_clean,"train")
    plot_boxplot(test_clean,"test")
    plot_boxplot(validation_clean,"validation")
    #draw wordclouds
    plot_wordclouds(train_clean)

