# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 02:55:50 2020

@author: wangk
"""


import numpy as np
import pandas as pd 
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


electronics_data=pd.read_csv("ratings_Electronics (1).csv",
                             names=['userId', 'productId','Rating','timestamp'])

#check data information
electronics_data.head()
#shape of data
electronics_data.shape
#taking subset of dataset
electronics_data=electronics_data.iloc[:1048576,0:]
#check data types
electronics_data.dtypes

electronics_data.info()

electronics_data.describe()['Rating'].T
#print max and min rating
print('Minimum rating is: %d' %(electronics_data.Rating.min()))
print('Maximum rating is: %d' %(electronics_data.Rating.max()))
#print missing values
print('Number of missing values across columns: \n',electronics_data.isnull().sum())
#draw rating distribution
with sns.axes_style('white'):
    g = sns.factorplot("Rating", data=electronics_data, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")


print("Total data ")
print("-"*50)
print("\nTotal no of ratings :",electronics_data.shape[0])
print("Total No of Users   :", len(np.unique(electronics_data.userId)))
print("Total No of products  :", len(np.unique(electronics_data.productId)))

#dropping the timestamp column
electronics_data.drop(['timestamp'], axis=1,inplace=True)
#analysis of rating given by user 
no_of_rated_products_per_user = electronics_data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)

no_of_rated_products_per_user.head()

no_of_rated_products_per_user.describe()

quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')

#draw relationship between no of rating by user and value at the quantile
plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()

print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )

#popularity based recommentation system implementation

new_df=electronics_data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)
no_of_ratings_per_product = new_df.groupby(by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()


new_df.groupby('productId')['Rating'].mean().head()

new_df.groupby('productId')['Rating'].mean().sort_values(ascending=False).head()

new_df.groupby('productId')['Rating'].count().sort_values(ascending=False).head()

ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())

ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['Rating'].count())

ratings_mean_count.head()

ratings_mean_count['rating_counts'].max()

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)

popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind = "bar")

#model based collaborative filtering system implementation

new_df1=new_df.head(10000)
ratings_matrix = new_df1.pivot_table(values='Rating', 
                index='userId', columns='productId', fill_value=0)
ratings_matrix.head()
ratings_matrix.shape

X = ratings_matrix.T
X.head()

X.shape

X1 = X

#Decomposing the Matrix
from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape

#Correlation Matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape

X.index[75]

i = "B00000K135"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID

correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape

Recommend = list(X.index[correlation_product_ID > 0.65])
# Removes the item already bought by the customer
Recommend.remove(i) 
print("The recommended products are:",Recommend[0:24])

