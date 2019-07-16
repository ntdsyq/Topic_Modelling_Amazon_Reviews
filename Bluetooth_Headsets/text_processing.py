# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:40:20 2019

@author: yanqi
"""

import os
from model_utils import detail_cat, proj_path
os.chdir(proj_path)

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from bs4 import BeautifulSoup
import nltk
from nltk import FreqDist
#nltk.download('stopwords') # run this line only once
from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer  # alternative lemmatizer
# lemma = WordNetLemmatizer()

import spacy
import re
import string
import pickle

from specialmaps import CONTRACTION_MAP
from collections import defaultdict
    
########## functions for text cleaning and processing ##########

def freq_words(x, terms = 30):
    # function to plot most frequent terms
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top n most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
  
def chk_has_html(rev_df,print_all=True):
    # check for review text that has tags such as &#34;
    # rev_df: e.g. df['reviewText']
    html_num = defaultdict(int)
    review_w_html = []
    for idx, text in enumerate(rev_df):
        strange_text = re.findall(r'&#\d+;',text)
        if len(strange_text) > 1:
            for t in strange_text:
                html_num[t] += 1
            review_w_html.append(text)
    
    if print_all == True:
        print(sorted(html_num.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))     
    return html_num

def chk_html_num(rev_df,tag):
    # prints reviews that contain the special HTML numbers specified by tag
    # tag: HTML numbers with pattern r'&#\d+;'
    for idx, text in enumerate(rev_df):
        if tag in text:
            print(idx, text, '\n')

def convert_html(rev): 
    # replace the html tags by the corresponding symbol using HTML_MAP
    html_entity = re.findall(r'&#\d+;',rev)
    rev_new = rev
    if len(html_entity) > 0:
        for t in html_entity:
            rev_new = rev_new.replace(t,HTML_MAP[t])
    return rev_new

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):    
    # expand contractions. I'm -> I am. Aren't -> are not
    # This is a simple version, see more elaborate version that considers word sense: https://pypi.org/project/pycontractions/
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_punctuation(rev, punc_char):
    # remove punctuation from the input string rev
    return ''.join([ch for ch in rev if ch not in punc_char])

def remove_stopwords(rev,stop_words):
    # rev is a review text, one string
    rev_new = " ".join([i for i in rev.split() if i not in stop_words])
    return rev_new

def lemmatization(rev, nlp): 
    # lemmatize input string rev, i.e. reduce all forms of a word to the lemma
    doc = nlp(" ".join(rev.split()))
    return " ".join([token.lemma_ for token in doc])

def lemmatization_tag(rev, nlp, tags = ['NOUN', 'ADJ']): # filter noun and adjective
    # meaning of POS tags: https://www.clips.uantwerpen.be/pages/MBSP-tags
    # compare lemmatization methods: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    doc = nlp(" ".join(rev.split()))
    return " ".join([token.lemma_ for token in doc if token.pos_ in tags])

def correct_dots(rev):
    # correct 1. spacing error e.g. "this is great.Recommend to anyone" 
    #         2. special punctuation wordA...wordB...wordC
    # replace any "." by " "
    # rev: a string
    # improvement: use regex 
    return " ".join(rev.replace("."," ").split())

def remove_num(s):
    # s is a string, e.g. an entire review text
    return ''.join([i for i in s if not i.isdigit()])

def chk_cleaned(df, cols, n = 1):
    # check the same text after each processing step
    rs = random.randint(1,101)
    for c in cols:
        print(df[c].sample(n, random_state = rs),'\n')


########## processing of review text starts here ##########
with open('df.pickle','rb') as f:
    df = pickle.load(f)
    
# use beautiful soup to remove html entities
# apply the function twice to completely strip off all the HTML entity numbers and tags
# use .loc to avoid setting with copy warning: https://www.dataquest.io/blog/settingwithcopywarning/
df.loc[:,'review_no_html'] = df['reviewText'].apply(remove_html_tags)
df.loc[:,'review_no_html'] = df['review_no_html'].apply(remove_html_tags)  

# expand contractions
df.loc[:,'review_no_contraction'] = df['review_no_html'].apply(expand_contractions)

# fix 1. spacing error e.g. "this is great.Recommend to anyone" 
#     2. special punctuation wordA...wordB...wordC
df.loc[:,'review_no_dots'] = df['review_no_contraction'].apply(correct_dots)

# remove punctuations
punc_char = set(string.punctuation)
df.loc[:,'review_no_punc'] = [remove_punctuation(r, punc_char) for r in df['review_no_dots']]

# remove stopwords, round 1
stop_words = stopwords.words('english')

# add stopwords that are unique to this collection of text
stop_words.extend(['headset','bluetooth','headphone','headphones'])  
#brandlist = ['Plantronics', 'Jabra', 'Motorola', 'LG', 'Kinivo', 'Samsung', 'Sony',
#       'JayBird', 'soundbot', 'iKross', 'Bose', 'MEElectronics', 'Jawbone']
#brandlist = [b.lower() for b in brandlist]
#stop_words.extend(brandlist)

df.loc[:,'review_no_stopwords'] = [remove_stopwords(r.lower(), stop_words) for r in df['review_no_punc']]
freq_words(list(df['review_no_stopwords']))

# lemmatize with SpaCy lemmatizer, this is a relatively slow preprocessing step 
nlp = spacy.load('en', disable=['parser', 'ner'])
df.loc[:,'review_lemmatized'] = df['review_no_stopwords'].apply(lemmatization, nlp=nlp)
freq_words(list(df['review_lemmatized']))  # consider remove use, phone, ear

# run remove stop words again because the lemmatization process generates more stop words
df.loc[:,'review_lemmatized'] = df['review_lemmatized'].apply(remove_stopwords, stop_words = stop_words)

# remove numbers
df.loc[:, 'review_lemmatized'] = df['review_lemmatized'].apply(remove_num)

# try stemming
#from nltk.stem import PorterStemmer
#porter = PorterStemmer()
#df.loc[:, 'review_stemmed'] = df['review_lemmatized'].apply(lambda x: [porter.stem(w) for w in x.split()])
# stemming did not work so well, chopped off a lot of "e" e.g. recycle -> recylc, replace, charge; 
# changed "y" to i, e.g. story -> stori

# check text after each processing step
cols = ['reviewText','review_no_html','review_no_contraction','review_no_dots','review_no_punc','review_no_stopwords','review_lemmatized']
chk_cleaned(df,cols = cols)

# save processed review df for topic modeling
save_cols = ['asin', 'brand', 'title', 'overall', 'summary', 
       'reviewYear', 'review_no_html', 'review_lemmatized']
df[save_cols].to_csv(detail_cat + "_processed.csv", index = False)

