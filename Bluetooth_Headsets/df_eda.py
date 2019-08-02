# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:49:28 2019

@author: yanqi
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_utils import detail_cat
pd.set_option('display.max_colwidth', -1) 
pd.set_option('display.max_columns', 500)

sns.set()
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

# eda of raw review, meta data for selected categories
# basic data cleaning without getting into processing of the reviewText: missing values, filter by time, number of reviews

def read_full_DF(file_prefix):
    df_rev = pd.read_csv(file_prefix + '_reviews.csv',index_col = 0)
    df_meta = pd.read_csv(file_prefix + '_meta.csv',index_col = 0)
    return df_rev, df_meta

def chk_mv(df):
    # check for missing values in each column of the dataframe df
    mv_bycol = pd.DataFrame( df.isnull().sum(axis=0), columns = ['num_mv'])
    mv_bycol['pct_mv'] = mv_bycol['num_mv']/df.shape[0]
    mv_bycol = mv_bycol.sort_values('num_mv', ascending=False)
    mv_by_col = mv_bycol[mv_bycol['num_mv'] > 0]
    print(mv_by_col)
    
def chk_product(s):
    # check if product name contains the string "headset", "headphone", or "earbud"
    keep = False
    for s0 in ['headset','headphone','earbud','earpiece', 'earset', 'earphone','headsets','headphones','earbuds','earpieces', 'earsets', 'earphones']:
        if s0 in s.lower():
            keep = True
            break
        else:
            continue
    return keep
    
df_rev, df_meta = read_full_DF(detail_cat)
print(df_rev.columns,'/n')
print(df_rev.dtypes,'/n')
print(df_rev.shape,'/n')

print(df_meta.columns,'/n')
print(df_meta.dtypes,'/n')
print(df_meta.shape,'/n')

dfall = df_rev.merge(df_meta, on = 'asin')
print(dfall.shape,'\n')
print(dfall.columns,'\n')
chk_mv(dfall)

# filter for products with brands, 69% reviews miss brand info, may need to infer from the title
from datetime import datetime
dfall.loc[:,'reviewYear'] = dfall['unixReviewTime'].apply(lambda x: datetime.utcfromtimestamp(x).year)
df = dfall[['asin','brand','title', 'overall','summary', 'reviewText', 'reviewYear']]


# time distribution
# df.reviewYear.value_counts()
# number of reviews by year and number of products by year
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
prod_by_yr = df[['reviewYear','asin']].drop_duplicates()
g2 = sns.countplot( x = 'reviewYear', data = prod_by_yr, color = "lightblue", ax = ax1 )
g2.set_xlabel('Year')
g2.set_ylabel('Number of Products Reviewed')

g1 = sns.countplot( x = 'reviewYear', data = df, color = "lightgreen", ax = ax2 )
g1.set_xlabel('Year of Review')
g1.set_ylabel('Number of Reviews')
plt.savefig("eda_rev_prod_byyear.pdf")

# filter for most recent 4 years
df = df.loc[ df.reviewYear >= 2011, :]

# Basic EDA 
# number of asins, number of associated reviews for each asin
review_cnts = df.groupby(['asin','title']).agg({'reviewText':'count'}).sort_values(by = 'reviewText', ascending = False)
review_cnts = review_cnts.reset_index()
review_cnts = review_cnts.rename({'reviewText':'reviewCnt'}, axis = 'columns')
review_cnts.shape[0]

#review_cnts = review_cnts.merge(df_meta[['asin','title']], on = 'asin')
topn = 20
print('Top most reviewed:\n', review_cnts.head(topn),'\n')
print(review_cnts.reviewCnt.describe())

# number of products with > 50 reviews each
print(np.sum(review_cnts.reviewCnt > 50))

# create a proxy for brand
df.loc[:,'brand_inferred'] = df['title'].apply(lambda x: str(x).split()[0])
df['brand'].fillna(df['brand_inferred'], inplace=True)
chk_mv(df)

from bs4 import BeautifulSoup
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

df['brand'] = df['brand'].apply(lambda s: remove_html_tags(s))
df['brand'].value_counts()

# drop records with missing reviewText or title
df = df.dropna(subset = ['reviewText','title'])
df['brand'].value_counts()

# filter title for headset, headphone, earbud
print(df['asin'].nunique())
keep_prod = df['title'].apply(chk_product)
print(np.sum(keep_prod))
print(df[ keep_prod == False ]['title'].nunique())
df = df[ keep_prod == True ]
print(df['asin'].nunique())

# fix row index 
df.reset_index(drop=True, inplace = True)
df.columns
df['asin'].head(5)

# save processed df to pickle file
import pickle
with open('df.pickle', 'wb') as f:
    pickle.dump(df, f)
