# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:42:34 2019

@author: yanqi
"""
## Implement various business cases using the product level topic counts produced in reviews_agg_topics_lda.py

import os
from model_utils import proj_path
os.chdir(proj_path)
os.getcwd()

import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from model_utils import load_processed_data
pd.set_option('display.max_columns', 500)

sns.set()
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

# load product (asin) level topic counts and product name, brand info
with open('agg_prod.pickle','rb') as f:
    [df_prod, df_prod_p] = pickle.load(f)

# load annotated topics    
topic_df = pd.read_csv("final_topics.csv")
nt = topic_df.shape[0]

# for specified asin, return topn topics with highest # of mentions in this product's reviews
def prod_topn_topics(asin, filter_topic = True, topn = 5):
    cols = [ "gt_prob" + str(i) for i in range(nt)]
    topic_prop = df_prod_p.loc[ df_prod_p.asin == asin, cols].transpose(copy = True)
    topic_prop.reset_index(drop = True, inplace = True)
    topic_prop.columns = ['prop_with_topic']
    # merge in topic info
    topic_prop = pd.merge(topic_df[['select', 'topic_num', 'name', 'meaning']],topic_prop, left_index = True, right_index = True)
    topic_prop.sort_values(by='prop_with_topic', ascending = False, inplace = True)
    
    # filter out the non-meaningful topics
    if filter_topic:
        topic_prop = topic_prop.loc[ topic_prop['select'] == 'yes', : ]
    
    # report topic index and topic name
    return topic_prop.iloc[0:topn, :].loc[:, ['topic_num', 'name', 'prop_with_topic']]

def prod_topprob_topics(asin, filter_topic = True, p0 = 0.08):
    cols = [ "gt_prob" + str(i) for i in range(nt)]
    topic_prop = df_prod_p.loc[ df_prod_p.asin == asin, cols].transpose(copy = True)
    topic_prop.reset_index(drop = True, inplace = True)
    topic_prop.columns = ['prop_with_topic']
    # merge in topic info
    topic_prop = pd.merge(topic_df[['select', 'topic_num', 'name', 'meaning']],topic_prop, left_index = True, right_index = True)
    topic_prop.sort_values(by='prop_with_topic', ascending = False, inplace = True)
    
    # filter out the non-meaningful topics
    if filter_topic:
        topic_prop = topic_prop.loc[ (topic_prop['select'] == 'yes') & (topic_prop['prop_with_topic'] > p0), : ]
    
    # report topic index and topic name
    return topic_prop[['topic_num', 'name', 'prop_with_topic']]

# get top 5 topics for the most highly reviewed 5 products
for i in range(5):
    print(df_prod_p[['asin', 'title', 'brand', 'overall','count_reviews', 'alltopics']].iloc[i,:])
    print(prod_topn_topics(asin = df_prod_p.iloc[i,:].asin),'\n')

# spot check a few others - jabra sport
print(df_prod_p.loc[ df_prod_p['asin'] == 'B005FVNHBI', ['asin', 'title', 'brand', 'overall','count_reviews', 'alltopics']])
print(prod_topn_topics(asin = 'B005FVNHBI'),'\n')  

# spot check a few others - plantronics, lower rating model
print(df_prod_p.loc[ df_prod_p['asin'] == 'B007MC6WGK', ['asin', 'title', 'brand', 'overall','count_reviews', 'alltopics']])
print(prod_topn_topics(asin = 'B007MC6WGK'),'\n')  # a plantronics with lower rating -> this is a more sporty version
# https://www.amazon.com/Plantronics-BackBeat-Bluetooth-Wireless-86800-03/dp/B007MC6WGK 
# topics overlap well with features highlighted on product page

# get top topics for the most highly reviewed 10 products based on a cut-off on % of reviews mentioning this topic
for i in range(10):
    print(df_prod_p[['asin', 'title', 'brand', 'overall','count_reviews', 'alltopics']].iloc[i,:])
    print(prod_topprob_topics(asin = df_prod_p.iloc[i,:].asin),'\n')

# top topics for all plantronics products, without filtering out topic 'plantronics'
brand = 'Motorola'
min_count_review = 50
for i in range(df_prod_p.shape[0]):
    if (df_prod_p.iloc[i,:].brand == brand) & (df_prod_p.iloc[i,:].count_reviews >= min_count_review):
        print(df_prod_p[['asin', 'title', 'brand', 'overall','count_reviews', 'alltopics']].iloc[i,:])
        print(prod_topprob_topics(asin = df_prod_p.iloc[i,:].asin),'\n')

# only look at products with at least 50 reviews, scatterplot of overall rating and amount of mention of great overall
df_prod_p_sub = df_prod_p.loc[ df_prod_p['count_reviews'] > min_count_review, :].copy()
plt.figure(figsize = (10,8))
sns.scatterplot(x = df_prod_p_sub['gt_prob8'], y = df_prod_p_sub['overall'])
plt.title('Amount of "great_overall" mention vs. overall rating')
plt.xlabel('Proportion of reviews with the topic "great_overall"')
plt.ylabel('Overall rating')
plt.savefig('Scatterplot of topic great_overall vs. overall rating.pdf')

# look for headsets that are most highly reviewed on the active life_style aspect

# clustering based on the proportions mentioning each topic on the products with at least min_num_review reviews
    