# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:56:31 2019

@author: yanqi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:47:52 2019

@author: yanqi
"""
import os
from model_utils import detail_cat, proj_path
#proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 4 - Capstone\\AmazonReview\\Bluetooth_Headsets'
os.chdir(proj_path)
import pandas as pd
pd.set_option('display.max_colwidth', -1)  # to view entire text in any column
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import string
import gzip
from pprint import pprint

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

os.environ['MALLET_HOME'] = "C:/Users/yanqi/Library/mallet-2.0.8"
mallet_path = "C:/Users/yanqi/Library/mallet-2.0.8/bin/mallet"

import pyLDAvis
import pyLDAvis.gensim   # for visualizing found topics
from model_utils import qc_dict, out_topics_docs, check_topic_doc_prob, topn_docs_by_topic, select_k

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)  # suppressing deprecation warnings when running gensim LDA

## prepare dictionary and corpus (DTM) for modeling
# useful attributes & methods: dictionary.token2id to get mapping, dictionary.num_docs 
df = pd.read_csv(detail_cat + "_processed.csv")
reviews = df['review_lemmatized'].copy()
reviews = reviews.apply(lambda x: x.split())

# Dictionary expects a list of list (of tokens)
dictionary = corpora.Dictionary(reviews)

# remove terms that appear in < 3 documents, memory use estimate: 8 bytes * num_terms * num_topics * 3
dictionary.filter_extremes(no_below=3)  

# number of terms
nd = dictionary.num_docs
nt = len(dictionary.keys())
print("number of documents", nd)
print("number of terms", nt)

qc_dict(dictionary)

# create document term matrix (corpus), it's a list of nd elements, nd = the number of documents
# each element of DTM (AKA corpus) is a list of tuples (int, int) representing (word_index, frequency)
DTM = [dictionary.doc2bow(doc) for doc in reviews]

## single run of lda model with 10 topics as a start, takes 50+s
%time ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=DTM, num_topics=20, id2word=dictionary)

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=reviews, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

## Optimze number of topics k based on coherence score 
limit = 21
start = 5
step = 1
model_lst, coherence_lst = select_k(dictionary, DTM, reviews, limit, start=start, step=step)

# plot coherence score as function of number of topics
x = range(start, limit, step)
plt.plot(x, coherence_lst)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_lst):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
## Pick two models to save
lda1 = model_lst[15]  # 15 -> 20 topics
lda2 = model_lst[10]  # 10 -> 15 topics

## Save the best models
import pickle
with open('ldamodels.pickle', 'wb') as f:
    pickle.dump([lda1, lda2, DTM, dictionary], f)






