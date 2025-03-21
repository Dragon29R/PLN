import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if name == 'main':
    columns =["ENTREGA","OUTROS","PRODUTO","CONDICOESDERECEBIMENTO","ANUNCIO"]
    results = pd.read_csv("results/results_extra.csv")
    for column in columns:
        print("Results for column: ", column)
        # Plot the bar graph using seaborn
        k=10
        results['VARIABLE'] = results["MODEL"].apply(lambda x: x[:-10]) + "\n" + results["DATASET"]
        top_k = results[results["TARGET"]==column].sort_values(by="ACCURACY",ascending=False).head(k)
        min_percentage = top_k["ACCURACY"].min()
        plt.figure(figsize=(17, 6))
        plt.ylim(min_percentage-0.03, 1)
        sns.barplot(x='VARIABLE', y='ACCURACY', data=top_k, palette='viridis')
        plt.xlabel('ACCURACY')
        plt.ylabel('VARIABLE')
        plt.title(f'Top {k} ACCURACY for {column}')
        plt.show()
