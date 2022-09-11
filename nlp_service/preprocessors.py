import sys
import pandas as pd 
import numpy as np
import pickle
from datetime import date
import logging
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from deeppavlov import build_model, configs

#import spacy
from gensim.models import Word2Vec

logging.basicConfig(format = '[%(asctime)s] [LEVEL:%(levelname)s] %(message)s',
                        datefmt = ('%Y-%m-%d %H:%M:%S'),
                        level = logging.INFO,
                        stream = sys.stdout)

#nlp = spacy.load("ru_core_news_sm")
nltk.download('stopwords')
stop = set(stopwords.words("russian"))
remove_words = ['не', 'нет', 'больше', 'всегда', 'другой', 'иногда', 'когда', 'что', 'можно']
for w in remove_words:
    stop.remove(w)

CONFIG_PATH = configs.spelling_correction.levenshtein_corrector_ru
try:
    spelling_model = build_model(CONFIG_PATH, download=False)
except:
    logging.FATAL("spelling_model failed to create!")

#w2v_model = Word2Vec(size=32, min_count=5, window=5).wv.load("./models/w2v.json")

def get_phrase_embedding(model, phrase):    
    vector = np.zeros([model.vector_size], dtype='float32')
    tokens = word_tokenize(phrase)
    emb_list = []
    for token in tokens:
        try:
            w = model.word_vec(token)
        except:
            w = np.zeros([model.vector_size], dtype='float32')
        emb_list.append(w)
    #weihted_emb = tfidf[idx].data * emb_list
    vector = np.mean(emb_list, axis=0)
    return vector

def get_weighted_embeddings(w2v_model, tfidf, df_col):
    weighted_emb = []
    features_train = tfidf.fit_transform(df_col)
    tfidf_feats = tfidf.get_feature_names()
    for i in range(0, len(df_col)):
        emb_list = []
        for token in word_tokenize(df_col[i]):
            try:
                w = w2v_model.word_vec(token)
            except:
                w = np.zeros([w2v_model.vector_size], dtype='float32')
            k = 0
            for idf_ix in features_train[i].indices:
                if tfidf_feats[idf_ix] == token:
                    w = w*features_train[i].data[k]
                k += 1
            emb_list.append(w)
        weighted_emb.append(np.mean(emb_list, axis=0))
    return weighted_emb

def spelling_correction(text):
    return spelling_model([text])[0]

def clear_text(string):
    words = []
    for word in string.split(" "):
        word = "".join(c for c in word if c.isalpha())
        words.append(word)
    string = " ".join([word for word in words])
    return " ".join(string.split())

def word_count(text):
    wc = 0
    for word in text.split(" "):
        wc += 1
    return wc

def fix_text(dataset, text_col):
    dataset[text_col] = dataset[text_col].apply(str)
    dataset[text_col] = dataset[text_col].apply(lambda x: x.lower())
    dataset['Stemmed_comment'] = dataset[text_col].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    dataset['Stemmed_comment'] = dataset[text_col].apply(lambda x: clear_text(x))
    #dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: x.replace("дс", "денежные средства"))
    dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: spelling_correction(x))
    stemmer = SnowballStemmer(language='russian')
    dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: ' '.join([stemmer.stem(i)  if i not in stop else '' for i in x.split()]))
    #dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: ' '.join([i.lemma_  if i not in stop else '' for i in nlp(x)]))
    dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: x.rstrip().lstrip())
    dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: re.sub(' +', ' ', x))
    dataset['Stemmed_comment'] = dataset['Stemmed_comment'].apply(lambda x: "empty" if x == "" else x)
    #dataset['Word_count'] = dataset['Stemmed_comment'].apply(lambda x: word_count(x))
    return dataset

def vectorize(df, vectorizer):
    features = vectorizer.transform(list(df['Stemmed_comment'].values)).toarray() #TODO refactor 'Stemmed_comment' hardcode
    return features

def prepare_data(train_dataset, test_dataset, cv = False, save_idf = False):
    if cv:
        print('Preparing data for cross-validation...')
    else:
        print('Preparing data for model training...')
    
    train_dataset = train_dataset[['Stemmed_comment', 'target']]
    test_dataset = test_dataset[['Stemmed_comment', 'target']]
    tfidf = TfidfVectorizer(ngram_range=(1, 2),min_df=0.001,max_df=0.999)
    features_train = tfidf.fit_transform(list(train_dataset['Stemmed_comment'].values)).toarray()
    features_test = tfidf.transform(list(test_dataset['Stemmed_comment'].values)).toarray()

    X_train = train_dataset.drop(['Stemmed_comment','target'],axis=1)
    X_train = np.hstack((X_train.values, features_train))
    #features_train = train_dataset.Stemmed_comment.apply(lambda x: get_phrase_embedding(w2v_model, x))
    #features_train = get_weighted_embeddings(w2v_model, tfidf, train_dataset.Stemmed_comment)
    #X_train = np.stack(features_train)
    y_train = train_dataset.target

    X_test = test_dataset.drop(['Stemmed_comment','target'],axis=1)
    X_test = np.hstack((X_test.values, features_test))
    #features_test = test_dataset.Stemmed_comment.apply(lambda x: get_phrase_embedding(w2v_model, x))
    #features_test = get_weighted_embeddings(w2v_model, tfidf, test_dataset.Stemmed_comment)
    #X_test= np.stack(features_test)
    y_test = test_dataset.target
    # TODO refactor save vectorizer
    if save_idf:
        with open('./models/vectorizer_{}.pk'.format(date.today().strftime("%d-%m-%Y")), 'wb') as fin:
            pickle.dump(tfidf, fin)
    if cv:
        #X = np.vstack((X_train,X_test))
        #y = pd.concat([y_train, y_test])
        print(f'Data shape: {X_train.shape}\nTarget shape: {y_train.shape}')
        print('Done.')
        return X_train, y_train
    else:
        return X_train, X_test, y_train, y_test
