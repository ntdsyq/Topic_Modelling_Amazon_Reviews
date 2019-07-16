# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:06:41 2019

@author: yanqi
"""
import os
from model_utils import detail_cat, proj_path
import pandas as pd
os.chdir(proj_path)
os.getcwd()
import pickle
from pprint import pprint
from model_utils import out_topics_docs, check_topic_doc_prob, topn_docs_by_topic
pd.set_option('display.max_columns', 500)

## Detailed look at the topic and associated documents 
## Interpret the topics, assign names & meaning


# load processed data
df = pd.read_csv(detail_cat + "_processed.csv")
reviews = df['review_lemmatized'].copy()
reviews = reviews.apply(lambda x: x.split())

# load model and gensim corpus
with open('ldamodels.pickle','rb') as f:
    lda, temp, x1, x2, DTM, dictionary = pickle.load(f) # chose model with 20 topics, selected 15 from 20

# quick look at topic keywords
nt = lda.num_topics
for t in range(nt):
    print(t, lda.show_topic(t))
    
# save topics for inspection
topic_df = pd.DataFrame(lda.show_topics(nt), columns=['topic_num','keywords'])
topic_df.to_csv("temp_topics.csv", index = False)

# check doc_topic probability distribution
len(lda[DTM])  # mallet produces a dense doc-topic probability matrix
topics_docs_dict = out_topics_docs(lda, DTM)  

doc_topic_probs = {}
for t in sorted(topics_docs_dict.keys()):
    dt_prob = check_topic_doc_prob(topics_docs_dict, t)
    print(dt_prob.describe(),"\n")
    
# examine each topic by topic key words, number of generated documents, document probabilities, docs with top probabilities
topic_num = 15

#print("topic", topic_num, "has", len(topics_docs_dict[topic_num]),"documents")
print("Distribution of probabilities of documents being generated from this topic:")
doc_prob = check_topic_doc_prob(topics_docs_dict, topic_num)
print(doc_prob.describe(),"\n")

top_docprobs = topn_docs_by_topic(topics_docs_dict,topic_num, 40)
idxs = pd.Series([x[0] for x in top_docprobs])
probs = pd.Series([x[1] for x in top_docprobs])
texts = pd.Series([df['review_no_html'][i] for i in idxs])
products = pd.Series([df['title'][i] for i in idxs])
asins = pd.Series([df['asin'][i] for i in idxs])
ratings = pd.Series([df['overall'][i] for i in idxs])
top_docs_df = pd.concat([asins, products, idxs, probs, ratings, texts], axis = 1)
top_docs_df.columns = ['asin','product','doc_id', 'prob_from_topic','rating','reviewText']

print(lda.show_topic(topicid=topic_num))
top_docs_df[['asin','product', 'prob_from_topic','rating']]
#top_docs_df['product']
top_docs_df['reviewText'][0:5]

# check for meaningful topics, combine similar topics, to arrive at final rating dimension
# go with 24 topics, prune to 18 topics
topic_df_final = topic_df.iloc[ [0, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19], :].copy()
topic_df_final.reset_index(drop=True, inplace=True)
topic_df_final.loc[:,'name'] = ['buttons','fit_head','noise_cancellation', 
                  'misc_issues_problems','cheap','phone_connection', 
                  'fit_ear','great_overall','sound_quality','charging','audio_performance','voice_command', 
                  'transaction_experience','active_lifestyle','battery']

topic_df_final.loc[:,'meaning'] = ["topic 0: various buttons e.g. power and volume control",
                  "topic 2: how the headeset fit on one's head and ear, such as with glasses",
                  "topic 3: whether the people you are talking to can hear background noise, especially in cars, whether has noise cancellation feature", 
                  "topic 4: miscellaneous problems and issues",
                  "topic 5: cheaply priced headsets (positive: good quality for the price; negative: breaks easy don't buy)",
                  "topic 6: connection between headset and phone, signal range, reconnection etc", 
                  "topic 7: fit on into/ear, due to e.g. earbud depth, cover, ear hook shape, ear wires",
                  "topic 8: highly satisfied overall, specially good quality for the price",
                  "topic 9: sound quality, sometimes with detailed evaluation, bass",
                  "topic 13: everything related to charging the headset: cables, charging speed, through car-charger, on-the-go charger",
                  "topic 14: experience (not just soundwise) listening to music, audiobooks, video",
                  "topic 16: Quality of voice command functionality/app for making calls and answering phones", 
                  "topic 17: Customer service experience related to shipping, return, replacement",
                  "topic 18: Usability in a active/sports setting, such as during running and gym workout",
                  "topic 19: Quality of battery e.g. talk time on full charge, battery indicator, deteriation over time"]

topic_df_final[['topic_num','name','meaning','keywords']].to_csv("final_topics.csv", index = False)

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


    


