# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:19:20 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 4 - Capstone\\AmazonReview\\'
os.chdir(proj_path)
import pandas as pd
import gzip

# functions for reading raw data (meta data for product info, and review data)
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDFn(path,n):
  # read the first n lines from the json.gz file
  i = 0
  df = {}
  parsed = parse(path)
  for d in parsed:
    print(d.keys())
    if i >= n:
        break
    else:
        df[i] = d
        i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def get_meta_DF_cat(path, select_cat):
  # read in a subset of the json.gz file filtered by the categories column 
  # select_cat is a list that represents a hierarchy of categories
  i = 0
  df = {}
  for d in parse(path):
    if select_cat in d['categories']:
        df[i] = d
        i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def get_review_DF_asin(path, asins):
  # get review data for the products with specified asins
  # asins: list of asins for products of interest
  i = 0
  df = {}
  for d in parse(path):
    if d['asin'] in asins:
        df[i] = d
        i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def write_full_DF(select_cat):
    # retrieved subset of meta and review data from raw data for select_category, save to csv file for faster access later
    product_type = select_cat[0].replace('&','and').replace(' ','_')
    
    # get meta data for the category of interest (indicated by select_cat)
    meta = get_meta_DF_cat("meta_" + product_type + ".json.gz", select_cat)
    detail_cat = select_cat[-1].replace(' ','_')
    meta.to_csv( detail_cat + '_meta.csv')

    # read in review data for products in the category of interest
    reviews = get_review_DF_asin('reviews_' + product_type + '_5.json.gz', list(meta.asin))
    reviews.to_csv( detail_cat + '_reviews.csv')

# generate csv files for meta and review data, for a selected category
# the output csv files will be under the root folder, should be copied to child folder for further analysis    
# select_cat = 'Cell Phones & Accessories->Accessories->Headsets->Bluetooth Headsets'.split('->')
select_cat = 'Cell Phones & Accessories->Accessories->Headsets->Bluetooth Headsets'.split('->')
write_full_DF(select_cat)