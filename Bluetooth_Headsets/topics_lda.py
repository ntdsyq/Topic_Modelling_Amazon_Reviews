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
from model_utils import out_topics_docs, check_topic_doc_prob, topn_docs_by_topic
pd.set_option('display.max_columns', 500)
## Detailed look at the topic and associated documents

# load processed data
df = pd.read_csv(detail_cat + "_processed.csv")
reviews = df['review_lemmatized'].copy()
reviews = reviews.apply(lambda x: x.split())

# load model and gensim corpus
with open('ldamodels.pickle','rb') as f:
    lda, lda2, DTM, dictionary = pickle.load(f)

len(lda[DTM])  # mallet produces a dense doc-topic probability matrix
topics_docs_dict = out_topics_docs(lda, DTM)  

# check doc_topic probability distribution
for t in sorted(topics_docs_dict.keys()):
    dt_prob = check_topic_doc_prob(topics_docs_dict, t)
    print(dt_prob.describe(),"\n")
    
# examine each topic by topic key words, number of generated documents, document probabilities, docs with top probabilities
topic_num = 19

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