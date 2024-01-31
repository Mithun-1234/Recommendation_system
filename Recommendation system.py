%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns


df = pd.read_excel("cloth.xlsx")
print(df.head())

df['Class Name'] = df['Class Name'].fillna(value=df['Class Name'].mode()[0])
df['Department Name'] = df['Department Name'].fillna(value=df['Department Name'].mode()[0])
df['Division Name'] = df['Division Name'].fillna(value=df['Division Name'].mode()[0])

total_depart_name=df['Department Name'].unique()
print(total_depart_name)

total_divi_name=df['Division Name'].unique()
print(total_divi_name)

total_class_name=df['Class Name'].unique()
print(total_class_name)

print(len(total_class_name))

df1 = df[['Review Text','Rating','Clothing ID']]
df1 = df1.dropna()
print(df1)

count = df.groupby("Clothing ID", as_index=False).count()
mean = df.groupby("Clothing ID", as_index=False).mean()
dfMerged = pd.merge(df, count, how='right', on=['Clothing ID'])
print(dfMerged)


dfProductReview = df.groupby("Clothing ID", as_index=False).mean()
print(dfProductReview.head(3))

ProductReviewSummary = df1.groupby("Clothing ID")["Review Text"].apply(str)
p = ProductReviewSummary.to_frame()
p['Review Text'] = p['Review Text'].str.replace('\d+'," ")
p['Review Text'] = p['Review Text'].str.replace('\n'," ")
p['Review Text'] = p['Review Text'].str.strip(" ")
p.shape[0]

p['Review Text'] = p['Review Text'].str.replace('@user',' ')
p['Review Text'] = p['Review Text'].str.replace('#',' ')
p['Review Text'] = p['Review Text'].str.replace("[^0-9a-zA-Z#]",' ')

print(df.isna().sum())

df1 = (df[df['Rating']=='5'])
data=[]
for i in df['Class Name'].unique():
  data.append([i,len(df[df['Class Name']==i])])



data=pd.DataFrame(data,columns=['Class Name','Rating'])
data = data.sort_values(by='Rating',ascending=False)
data.plot(x='Class Name',y='Rating',kind='bar',figsize=(10,5))

print(data)


#most no of rating using customer id
data3 =[]
for 	dress in df['Clothing ID'].unique():
  data3.append([dress,len(df[df['Clothing ID']==	dress])])



data3=pd.DataFrame(data,columns=['Clothing ID','Class Name'])
data3 = data.sort_values(by='Class Name',ascending=False)

print(data3)

df['Class Name'] = df['Class Name'].fillna(value=df['Class Name'].mode()[0])
df['Department Name'] = df['Department Name'].fillna(value=df['Department Name'].mode()[0])
df['Division Name'] = df['Division Name'].fillna(value=df['Division Name'].mode()[0])

print(df.isna().sum())

df2 = df[['Review Text','Title']]
df2 = df2.dropna()
print(df2)

X=df.iloc[:,:2]
Y=df.iloc[:,5]

X_train, X_test,Y_train,Y_train  = train_test_split(
    X,Y, test_size=0.25, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(p['Review Text'])
print((tfidf_matrix.shape))


cosine_similarities = cosine_similarity(tfidf_matrix,Y=None,dense_output=False)
cnum = (cosine_similarities.toarray())
print(((cosine_similarities[0][:1,:-19])))
type(cosine_similarities)


#Recommendation
output=[]

import tkinter as tk
from tkinter import *
from tkinter import ttk


root = tk.Tk()
root.geometry("700x550")
root.title("Recommendation System")

def get_recommendations(id):

    print("the product selected is ",id)


    a = cosine_similarities.getcol(id)

    val = list(enumerate(a.data))

    

    b= dict(val)

    print(b)

    c = sorted(b.items(),key=lambda x:x[1],reverse=True)[1:16]

    k = 1
  
    for idx in c:
        z=p.index[idx[0]]
        print("The {} Recommendation is item {}".format(k,z))
        output.append(z)
        k += 1
        
my_label = tk.Label(root, text ='Product Id', font=("calibre",20))
my_label.pack(pady=20)  
root.configure(background='light yellow')

        
def submit():
    greet = get_recommendations(int(my_box.get()))
    my_label.config(text=greet)

    new= Toplevel(root)
    new.geometry("750x550")
    new.title("Recommended Items")
    Label(new, text="The Recommended Products are \n "+ str(output), font=('Helvetica 12')).pack(pady=20)
    new.configure(background='light blue')
    
my_box = tk.Entry(root)
my_box.pack(pady=20)


my_button = tk.Button(root, text="Submit",command=submit)
my_button.pack(pady=20)



root.mainloop()


























