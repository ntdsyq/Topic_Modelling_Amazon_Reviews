# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:49:28 2019

@author: yanqi
"""
import numpy as np
import pandas as pd
import os
from model_utils import detail_cat, proj_path
os.chdir(proj_path)
pd.set_option('display.max_colwidth', -1) 
pd.set_option('display.max_columns', 500)


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
df.reviewYear.value_counts()

# filter for most recent 4 years
df = df.loc[ df.reviewYear >= 2011, :]

# 
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

# fix row index 
df.reset_index(drop=True, inplace = True)
df.columns
df['asin'].head(5)

# save processed df to pickle file
import pickle
with open('df.pickle', 'wb') as f:
    pickle.dump(df, f)
