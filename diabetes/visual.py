import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visual():
    df = pd.read_csv('E:\dev\MLP_c02\diabetes.csv')

    pd.set_option('display.max_columns', 9)  
    print(df)

    df.hist(figsize=(6,7))
    plt.show()

    plt.subplots(3,3,figsize=(8,9))
    for idx ,col in enumerate(df.columns):
        ax=plt.subplot(3,3,idx+1)
        ax.yaxis.set_ticklabels([])
        sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel= False, kde_kws={'linestyle':'-', 'color':'black', 'label':"No Diabetes"})
        sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel= False, kde_kws={'linestyle':'--', 'color':'black', 'label':"No Diabetes"})
        ax.set_title(col)

    plt.subplot(3,3,9).set_visible(False)
    plt.show()


