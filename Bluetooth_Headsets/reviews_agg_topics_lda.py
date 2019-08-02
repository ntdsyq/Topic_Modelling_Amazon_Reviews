# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:48:21 2019

@author: yanqi
"""
## Various diagnostics of the document_topic probabilities (one document = one review)
## Choose threshold for calling "a topic is present in a review"
## Aggregate topic counts at product level for building business use cases
## outputs: 
### - A few PDF charts 
### - Summary in "review topic probability diagnositics.xlsx"

from model_utils import detail_cat
import pandas as pd

import pickle
from pprint import pprint
from model_utils import out_topics_docs, check_topic_doc_prob, topn_docs_by_topic, load_processed_data
pd.set_option('display.max_columns', 500)

import gensim
from gensim import corpora
import os
os.environ['MALLET_HOME'] = "C:/Users/yanqi/Library/mallet-2.0.8"
mallet_path = "C:/Users/yanqi/Library/mallet-2.0.8/bin/mallet"

# load review data, lda model and gensim corpus
df, reviews = load_processed_data()
with open('ldamodels.pickle','rb') as f:
    lda, temp, x1, x2, DTM, dictionary = pickle.load(f) # chose model with 20 topics, selected 15 from 20


# load processed topics
nt = lda.num_topics
topic_df = pd.read_csv("final_topics.csv")
#topic_df = pd.DataFrame(lda.show_topics(nt), columns=['topic_num','keywords'])
topics_docs_dict = out_topics_docs(lda, DTM)  

## Check document probability distributions for each topic
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np
plt.style.use('ggplot')

doc_topic_probs = {}
for t in sorted(topics_docs_dict.keys()):
    doc_topic_probs[t] = check_topic_doc_prob(topics_docs_dict, t)
doc_topic_prob_df = pd.DataFrame(doc_topic_probs)

sns.set(style = 'white')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
for i, topic_num in enumerate(topic_df['topic_num']):
    g1 = sns.kdeplot(doc_topic_probs[topic_num], ax = ax1)
    g2 = sns.kdeplot(doc_topic_probs[topic_num], ax = ax2, cumulative = True, label = topic_df['name'][i])
g1.set_xlabel('log(probability of document from topic)')
g1.set_ylabel('Density')
g2.set_xlabel('probability of document from topic')
g2.set_ylabel('Cumulative probability')
plt.savefig("document_topic_probability_distribution_all.pdf")

# distribution of length of the review
rev_length = reviews.apply(len)
rev_length_deciles = pd.qcut(rev_length,10)
plt.figure(figsize = (10,8))
sns.distplot(rev_length)
plt.xlabel("number of tokens in cleaned review text")
plt.ylabel("probability density")
plt.savefig("review_length_distribution.pdf")

# scatterplot of the length of the review vs. probability from each topic
topic_num = 0
plt.figure(figsize = (10,8))
sns.scatterplot(x = np.log10(rev_length), y = doc_topic_probs[topic_num])
plt.xlabel("Length of review on log scale")
plt.ylabel("Probability of review from topic" + str(topic_num))
plt.savefig("scatterplot of log review length vs topic probability for topic" + str(topic_num) + ".pdf")

# violinplot of doc_topic probabilities grouped by rev_length_deciles
plt.figure(figsize = (15,8))
sns.violinplot(x= rev_length_deciles, y= np.log(doc_topic_probs[topic_num]), palette="Set3")
plt.xlabel('Review Length Deciles')
plt.ylabel('Probability from topic' + str(topic_num))
plt.savefig("Violinplot of review length decile vs topic probability for topic" + str(topic_num) + ".pdf")

# boxplot of doc_topic probabilities grouped by rev_length_deciles
plt.figure(figsize = (15,8))
sns.boxplot(x= rev_length_deciles, y= np.log(doc_topic_probs[topic_num]), palette="Set3")
plt.xlabel('Review Length Deciles')
plt.ylabel('Probability from topic' + str(topic_num))
plt.savefig("Boxplot of review length decile vs topic probability for topic" + str(topic_num) + ".pdf")

# number of reviews that exceed a threshold, as a function of length of review
prob0 = 0.09
gt_prob = {}
for t in range(len(doc_topic_probs)):
    gt_prob["gt_prob"+str(t)] = doc_topic_probs[t] > prob0
has_topic_df = pd.DataFrame(gt_prob)  # indicator matrix for whether a review contains a topic

revlen_numtopics_df = pd.concat([rev_length_deciles, has_topic_df], axis = 1)
revlen_numtopics_df.rename({"review_lemmatized":"rev_length_deciles"}, axis = 'columns', inplace = True)
cnttopics_bydecile_df = revlen_numtopics_df.groupby('rev_length_deciles').agg(sum)
#cnttopics_bydecile_df.to_csv("num_of_topics_by_rev_length_deciles.csv")  # number of reviews containing each topic, by review length deciles

for i in range(cnttopics_bydecile_df.shape[1]):
    print(i, "th topic", cnttopics_bydecile_df.iloc[:,i], "\n")
    
# Number of topics in each review
cnt_bytopics = has_topic_df.sum(axis = 1)
cnt_bytopics.describe()  # more than 50% reviews has 0 or 1 topics
cnt_bytopics.unique()  # each review has 0 to 5 topics

# Number of topics vs. # of reviews containing that number of topics
numrev_bynumtopic = cnt_bytopics.value_counts().reset_index()
numrev_bynumtopic.rename({"index":"number_of_topics",0:"count_reviews"}, axis = "columns", inplace = True)
numrev_bynumtopic.loc[:,'pct_reviews'] = numrev_bynumtopic['count_reviews']/reviews.shape[0]
numrev_bynumtopic.sort_values(by = 'number_of_topics',inplace = True)
#numrev_bynumtopic.to_csv("num_of_reviews_by_num_of_topics.csv")

# write the counts to excel file
with pd.ExcelWriter('review topic probability diagnositics.xlsx') as writer:
    cnttopics_bydecile_df.to_excel(writer, sheet_name='numtopics_by_revlengthdeciles')
    numrev_bynumtopic.to_excel(writer, sheet_name='numreviews_by_numtopics',index = False)
    
# length of review vs. number of topics in review
# longer reviews are more likely to contain multiple topics, but in some long reviews, only 1 topic is detected
plt.figure(figsize = (15,8))
sns.boxplot(x = cnt_bytopics, y = rev_length)
plt.xlabel("number of topics present in each review")
plt.ylabel("count of reviews with specified number of topics")
plt.savefig("boxplot_numtopics_numreviews.pdf")

# check the longest reviews with only 1 topic
df.loc[:,'length'] = rev_length
df.loc[:,'num_topics'] = cnt_bytopics

# spot check the long reviews with only one topic
# idx = 1: about tech support customer service from plantronics
# idx = 2: raved about battery life, but has some other topics covered as well
# idx = 3: good amount on sound quality, but covered a range of topics
chk_revs1 = df.loc[ (df.length > 500) & (df.num_topics == 1) & (df.length < 1000) , 'review_no_html']  
idx = chk_revs1.index[3]
print(chk_revs1[idx])
print(has_topic_df.iloc[idx,:])
print(topic_df['name'][np.argwhere(has_topic_df.iloc[idx,:] == True)[0][0]])

# spot check the reviews with 5 topics
# idx = 0: len = 320. captured jabra, buttons, charging and voice_command. 
# idx = 1: len = 250. active lifestyle is the main topic, the other topics not as prominent
# idx = 2: len = 322. active lifestyle, ear and head fit, jabra, connection. 
chk_revs2 = df.loc[ df.num_topics == 5, 'review_no_html' ]
idx = chk_revs2.index[2]
print(chk_revs2[idx])
print(df['length'][idx])
print(has_topic_df.iloc[idx,:])
topic_idx = [item for sublist in np.argwhere(has_topic_df.iloc[idx,:] == True) for item in sublist]
print(topic_df['name'][topic_idx])

# check reviews not super short but has 0 topic detected
# mixed, sometimes  lower threshold 0.08 or 0.07 will return meaningful topics, sometimes truly no topics
chk_revs3 = df.loc[ (df.length > 50) & (df.num_topics == 0) & (df.length < 60) , 'review_no_html']
idx = chk_revs3.index[3]
print(chk_revs3[idx])
print(df['length'][idx])
print(doc_topic_prob_df.iloc[idx,:])
print(topic_df['name'])

# conclusion: 
# some reviews extremely long and talk about everything, topic probabilities disperse too much, may only get 1 topic 
# median length reviews are more crisp, several topics can be detected
# very short reviews may not meet threshold for even one topic, though they may have covered several topics 
# can force each review to have at least one topic

# for each product, record how many topics, and how many times each topic is talked about
df['brand'].value_counts()
df.loc[df.brand == 'Plantronics'].asin.value_counts()
df.loc[df.asin == 'B005IMB5NG'].title.value_counts()

# number of reviews by asin
revcnt_byasin = df['asin'].value_counts().reset_index()
revcnt_byasin.rename({"index":"asin", "asin":"count_reviews"}, axis = "columns", inplace = True)

# pick case study for 2 to 3 products (with different overall ratings)
df_prod = pd.concat([df['asin'],has_topic_df], axis = 1).groupby('asin').agg('sum').reset_index()
df_prod = df_prod.merge(revcnt_byasin, on = 'asin')

# alltopics may be greater than count_reviews, as some reviews will cover multiple topics
df_prod.loc[:,'alltopics'] = df_prod.iloc[:,1:21].sum(axis = 1)  
df_prod.head(1)

# merge in product title and brand information
# easier way to get counts of unique combinations of columns: df_prod.groupby(['asin','title']).size().reset_index(name = 'Freq')
prod_info = df[['asin','title','brand']].drop_duplicates()
avg_rating = df.groupby(['asin'])[['overall']].agg('mean').reset_index()
prod_info=prod_info.merge(avg_rating, on = 'asin')
prod_info.columns
df_prod = prod_info.merge(df_prod, on = 'asin')
df_prod.sort_values(by = 'count_reviews', ascending = False, inplace = True )

# percentage of all reviews mentioning a topic
df_prod_p = df_prod.copy()
for i in range(nt):
    df_prod_p[ "gt_prob"+str(i) ] = df_prod_p[ "gt_prob"+str(i) ]/df_prod_p['count_reviews']


# write the counts to excel file for visual inspection
with pd.ExcelWriter('review topic probability diagnositics.xlsx') as writer:
    cnttopics_bydecile_df.to_excel(writer, sheet_name='numtopics_by_revlengthdeciles')
    numrev_bynumtopic.to_excel(writer, sheet_name='numreviews_by_numtopics',index = False)
    df_prod.to_excel(writer, sheet_name = 'counts_byasin', index = False)
    df_prod_p.to_excel(writer, sheet_name = 'counts_p_byasin', index = False)
    

# write to pickle file for later use
with open('agg_prod.pickle', 'wb') as f:
    pickle.dump([df_prod, df_prod_p], f)