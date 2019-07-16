# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:48:21 2019

@author: yanqi
"""

import os
from model_utils import detail_cat, proj_path
import pandas as pd
os.chdir(proj_path)
os.getcwd()
import pickle
from pprint import pprint
from model_utils import out_topics_docs, check_topic_doc_prob, topn_docs_by_topic, load_processed_data
pd.set_option('display.max_columns', 500)

## Detailed look at the topic and associated documents 
## Interpret the topics, assign names & meaning
df, reviews = load_processed_data()

# load model and gensim corpus
with open('ldamodels.pickle','rb') as f:
    lda, temp, x1, x2, DTM, dictionary = pickle.load(f) # chose model with 20 topics, selected 15 from 20

# quick look at topic keywords
nt = lda.num_topics
topic_df = pd.DataFrame(lda.show_topics(nt), columns=['topic_num','keywords'])
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
for i, topic_num in enumerate(topic_df_final['topic_num']):
    g1 = sns.kdeplot(doc_topic_probs[topic_num], ax = ax1)
    g2 = sns.kdeplot(doc_topic_probs[topic_num], ax = ax2, cumulative = True, label = topic_df_final['name'][i])
g1.set_xlabel('log(probability of document from topic)')
g1.set_ylabel('Density')
g2.set_xlabel('probability of document from topic')
g2.set_ylabel('Cumulative probability')
plt.savefig("document_topic_probability_distribution.pdf")

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
sns.scatterplot(x = np.log10(rev_length), y = doc_topic_probs[topic_num])
plt.xlabel("Length of review on log scale")
plt.ylabel("Probability of review from topic" + str(topic_num))
plt.savefig("scatterplot of log review length vs topic probability for topic" + str(topic_num) + ".pdf")

# violinplot of doc_topic probabilities grouped by rev_length_deciles
plt.figure(figsize = (15,8))
sns.violinplot(x= rev_length_deciles, y= np.log(doc_topic_probs[topic_num]), palette="Set3")
plt.xlabel('Review Length Deciles')
plt.ylabel('Probability from topic' + topic_num)

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
gt_prob_df = pd.DataFrame(gt_prob)

revlen_numtopics_df = pd.concat([rev_length_deciles, gt_prob_df], axis = 1)
revlen_numtopics_df.rename({"review_lemmatized":"rev_length_deciles"}, axis = 'columns', inplace = True)
sum_df = revlen_numtopics_df.groupby('rev_length_deciles').agg(sum)
sum_df.to_csv("num_of_topics_by_rev_length_deciles.csv")

for i in range(sum_df.shape[1]):
    print(i, "th topic", sum_df.iloc[:,i], "\n")