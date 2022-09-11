import pandas as pd 
import numpy as np
import pickle
import re

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from .model import OverModel, SingleModel

from sklearn.feature_extraction.text import TfidfVectorizer


stop = set(stopwords.words("russian"))

def load_models(models_path):
    model = OverModel()
    model.load(models_path + "sklearn/", models_path + "vectorizer.pk")
    return model

def load_single_models(models_path):
    host_model = SingleModel()
    host_model.load(models_path + "host_ceo_clf_v0.1_100220121.cbm", models_path + "host_vectorizer.pk")
    no_oper_model = SingleModel()
    no_oper_model.load(models_path + "no_oper_ceo_clf_v0.1_100220121.cbm", models_path + "host_vectorizer.pk") # TODO add no_oper vectorizer later!!!
    return host_model, no_oper_model

def fix_text(comment):
    stop = set(stopwords.words("russian"))
    comment = comment.lower()
    tokenizer = RegexpTokenizer(r"\w{3,}")
    stemmed_comment = " ".join([word for word in tokenizer.tokenize(comment)])
    stemmer = SnowballStemmer(language='russian')
    stemmed_comment = ' '.join([stemmer.stem(i)  if i not in stop else '' for i in stemmed_comment.split()])
    stemmed_comment = stemmed_comment.rstrip().lstrip()
    stemmed_comment = re.sub(' +', ' ', stemmed_comment)
    return stemmed_comment

def postprocessing(comment, predictions, predictions_probability, class_labels, models_path):
    cls_prob = {"no_oper": 0.90, "remote_reboot": 0.94, "remote_balance": 0.94, "cards": 0.94, "host": 0.94, "cassette": 0.93, "empty": 0.95}
    
    classes = []
    class_probas = []
    for i in range(0, len(class_labels)):
        if predictions_probability[i][0][1] > cls_prob[class_labels[i]]:
            classes.append(class_labels[i])
            class_probas.append(predictions_probability[i][0][1])
    if ('CASSETTE TYPE' in comment) | ('cassette type' in comment):
        classes.append('cassette')
        class_probas.append(1.0)
    if classes == []:
        classes.append('empty')
        class_probas.append(1.0)
    resp = dict(zip(classes, class_probas))
    return single_models_processing(comment, resp, models_path)

def single_models_processing(comment, classes_dict, models_path):
    host_model, _ = load_single_models(models_path)
    # no_oper postprocessing
    bna_keywords = ['прием', 'внесен', 'приём', 'внести', 'приним', 'при']
    ch_keywords = ['выдач', 'снят', 'выдает', 'выдаёт']
    pay_keywords = ['платеж', 'оплат']
    if 'no_oper' in list(classes_dict.keys()):
        # TODO change code below
        if any(word in comment for word in bna_keywords) & (any(word in comment for word in ch_keywords) == False):
            classes_dict['no_oper_BNA'] = classes_dict.pop('no_oper')
        elif any(word in comment for word in ch_keywords) & (any(word in comment for word in bna_keywords) == False):
            classes_dict['no_oper_CH'] = classes_dict.pop('no_oper')
        elif any(word in comment for word in bna_keywords) & any(word in comment for word in ch_keywords):
            pass
        elif any(word in comment for word in pay_keywords):
            classes_dict['no_oper_payment'] = 1.0
    # host postprocessing
    if 'host' in list(classes_dict.keys()):
        h_class, _ = host_model.predict(comment)
        classes_dict[h_class[0]] = classes_dict.pop('host')
    return classes_dict
