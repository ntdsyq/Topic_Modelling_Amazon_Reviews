# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:06:41 2019

@author: yanqi
"""
## Detailed look at the topic and associated documents 
## Interpret the topics, assign names & meaning
## Select topics that are most meaningful (column 'select' in topic_df)

import os
from model_utils import detail_cat, proj_path
import pandas as pd
os.chdir(proj_path)
os.getcwd()
import pickle
from pprint import pprint
from model_utils import out_topics_docs, check_topic_doc_prob, topn_docs_by_topic, load_processed_data
pd.set_option('display.max_columns', 500)


df, reviews = load_processed_data()

# load model and gensim corpus
with open('ldamodels.pickle','rb') as f:
    lda, temp, x1, x2, DTM, dictionary = pickle.load(f) # chose model with 20 topics, selected 15 from 20

# quick look at topic keywords
nt = lda.num_topics
for t in range(nt):
    print(t, lda.show_topic(t))
    
# save topics for inspection
topic_df = pd.DataFrame(lda.show_topics(nt), columns=['topic_num','keywords'])
topic_df.to_csv("initial_topics.csv", index = False)

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
# go with 20 topics, 15 topics out of 20 are good
#topic_df_final = topic_df.iloc[ [0, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19], :].copy()
#topic_df_final.reset_index(drop=True, inplace=True)
topic_df.loc[:,'name'] = ['buttons','motorola_misc','fit_head','noise_cancellation', 
                  'misc_issues_problems','cheap','phone_connection', 
                  'fit_ear','great_overall','sound_quality','plantronics','not_sure','jabra',
                  'charging','audio_performance','not_sure','voice_command', 
                  'transaction_experience','active_lifestyle','battery']

topic_df.loc[:,'meaning'] = ["topic 0: various buttons e.g. power and volume control",
                  "topic 1: for or related to motorola products",
                  "topic 2: how the headeset fit on one's head and ear, such as with glasses",
                  "topic 3: whether the people you are talking to can hear background noise, especially in cars, whether has noise cancellation feature", 
                  "topic 4: miscellaneous problems and issues",
                  "topic 5: cheaply priced headsets (positive: good quality for the price; negative: breaks easy don't buy)",
                  "topic 6: connection between headset and phone, signal range, reconnection etc", 
                  "topic 7: fit on into/ear, due to e.g. earbud depth, cover, ear hook shape, ear wires",
                  "topic 8: highly satisfied overall, specially good quality for the price",
                  "topic 9: sound quality, sometimes with detailed evaluation, bass",
                  "topic 10: (primarily) high praises for the plantronics brand",
                  "topic 11: not sure about meaning",
                  "topic 12: for the jabra brand",
                  "topic 13: everything related to charging the headset: cables, charging speed, through car-charger, on-the-go charger",
                  "topic 14: experience (not just soundwise) listening to music, audiobooks, video",
                  "topic 15: not sure about meaning",
                  "topic 16: Quality of voice command functionality/app for making calls and answering phones", 
                  "topic 17: Customer service experience related to shipping, return, replacement",
                  "topic 18: Usability in a active/sports setting, such as during running and gym workout",
                  "topic 19: Quality of battery e.g. talk time on full charge, battery indicator, deteriation over time"]

topic_df.loc[:,"select"] = "no"
good_cols = [0, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19]
topic_df.iloc[good_cols, -1] = "yes"

topic_df[['select','topic_num','name','meaning','keywords']].to_csv("final_topics.csv", index = False)




    


