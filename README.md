### Project overview
Analyzing Amazon reviews with natural language processing techniques to uncover key topics (product features) discussed across a category of products. Topic modeling with Latent Dirichlet Allocation applied to reviews for bluetooth headsets. Topics discovered corresponded to specific product characteristics (e.g. "voice command function", "sound quality") and holistic assessments ("great overall", "cheap", and "use in active lifestyle"). Output can be used for such business cases as:
- By customers: prioritizing products with relevant features  
- By seller / manufacturer: monitor product perception to improve design and service

Read more in this [blogpost](https://nycdatascience.com/blog/student-works/learning-category-wise-product-features-from-amazon-reviews/)

### File organization
- process_raw.py: extracts review and meta data from the raw json.gz files for a category of interest, e.g. Bluetooth speakers
- Bluetooth_Headsets: main analyses folder. The pipeline consists of the following 6 steps  
    1 df_eda.py: EDA of raw data and basic data cleaning (missing value, filtering) without advanced text processing.
    2 text_processing.py: extensive text processing including expanding contractions, removing punctuations and stopwords, lemmatization and tokenization.
    3 train_lda.py: training LDA model and optimize parameters with the cleaned review text data
    4 topics_interpret_lda.py: interpret topics based on topic keywords, and reviews with the highest probability being generated from each topic. 
    5 reviews_agg_topics_lda.py: diagnostics of the document_topic probabilities. Aggregate topic counts at the review and product level.
    6 product_usecases.py: Implement various business cases using the product level topic counts produced in the previous step 
- reviews_LDA_TVmount.ipynb: notebook for initial analysis with TV mounts, an experiment bench and precursor to the 6-step pipeline above.
