
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — preprocessing, model inference and recommendation utilities.

This module provides helpers to normalize and lemmatize text, load
serialized model artifacts, predict sentiment and produce product
recommendations. Designed to be imported by an application or service.

Metadata and simple logger are configured here for consistent output.

__author__ = Dinesh Namdev
__batch__ C68
__email__ = dinesh.namdev@hotmail.com
__version__ = 1.0.0
__date__ = 11/4/2025
"""

# Import necessary libraries
from nltk.corpus.reader import reviews  # reader for NLTK corpora; provides access to 'reviews' corpus if available
import pandas as pd                     # pandas for DataFrame operations and CSV I/O
import re, nltk, spacy, string          # re for regex, nltk for NLP utilities, spacy for advanced NLP, string for string constants
import en_core_web_sm                    # spaCy small English model (alternative way to load the model)
import pickle as pk                      # pickle module (aliased as pk) for loading/saving serialized objects

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer  # TF-IDF transformer and count vectorizer classes
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier  # example classifiers (imported but not used in this file)
from nltk.corpus import stopwords        # stopwords corpus from NLTK (common words to remove)
from nltk.tokenize import word_tokenize  # tokenizer from NLTK for splitting text into words
from nltk.stem import LancasterStemmer   # stemming algorithm implementation
from nltk.stem import WordNetLemmatizer  # lemmatizer that uses WordNet

nltk.download('punkt')                   # ensure the punkt tokenizer models are downloaded (used by word_tokenize)
nltk.download('stopwords')               # ensure stopwords corpus is downloaded
nltk.download('wordnet')                 # ensure WordNet (required by WordNetLemmatizer) is downloaded
nltk.download('omw-1.4')                 # ensure the Open Multilingual Wordnet data is downloaded for lemmatization support

# load the pickle files 
count_vector = pk.load(open('pickle_file/count_vector.pkl','rb'))            # Count Vectorizer
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl','rb'))  # TFIDF Transformer
model = pk.load(open('pickle_file/Logistic_RegressionClassifier.pkl','rb'))  # Classification Model (Linear Regression Classifier)
recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl','rb'))   # User-User Recommendation System 

# Load spaCy's small English model. Disabling the named entity recognizer (ner) and parser
# can speed up processing if only tokenization/lemmatization/pos-tagging is required.
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Load the product dataset from CSV into a pandas DataFrame.
# Assumes 'sample30.csv' is available in the current working directory and uses comma as delimiter.
product_df = pd.read_csv('sample30.csv', sep=",")


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    """Remove the special Characters"""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

# text normalization functions
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

# remove punctuation and special characters
def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

# stopword list
stopword_list= stopwords.words('english')

# remove stop words
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

# stemming words
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

# lemmatizing verbs
def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

# normalizing the text
def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

# lemmatizing the text
def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

#predicting the sentiment of the product review comments
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output

# Normalize and Lemmatize the input text
def normalize_and_lemmaize(input_text):
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)

#Recommend the products based on the sentiment from model
def recommend_products(user_name):
    recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl','rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name','reviews_text']]
    output_df['lemmatized_text'] = output_df['reviews_text'].map(lambda text: normalize_and_lemmaize(text))
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df
    
# Get the top 5 products with positive sentiment
def top5_products(df):
    total_product=df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name','predicted_sentiment']).agg('count')
    rec_df=rec_df.reset_index()
    merge_df = pd.merge(rec_df,total_product['reviews_text'],on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x']/merge_df['reviews_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] ==  1][:5])
    return output_products