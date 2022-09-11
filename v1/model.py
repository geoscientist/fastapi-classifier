import os
import pickle
import pandas as pd 
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class OverModel:
    def __init__(self, model_arr = [], mod_cl = []):
        self.models = model_arr
        self.models_classes = mod_cl
        self.consistency = bool(len(self.models) == len(self.models_classes))
        self.vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.99, min_df=0.05)


    def info(self):
        for cl, model in enumerate(self.models, 0):
            print('---------------------------------------')
            print('class: ',self.models_classes[cl])
            print(model)
            print('---------------------------------------')

    
    def fit(self, train_data, train_labels, verbose = True):
        for cl, model in enumerate(self.models):
            if verbose: print(f'model class: {self.models_classes[cl]}, training...')
            current_target = train_labels.apply(lambda x: int(self.models_classes[cl] in x.split(','))).values
            model.fit(train_data, current_target)
            if verbose: print('Done.')

    def predict(self, comment):
        predictions = []
        predictions_probability = []
        features = self.vect.transform([comment]).toarray()
        for model in self.models:
            pred = model.predict(features)
            pred_prob = model.predict_proba(features)
            predictions.append(pred)
            predictions_probability.append(pred_prob)
        return predictions, predictions_probability, self.models_classes

    def save(self, fname):
        for cl, model in enumerate(self.models):
            pickle.dump(model, open(str(self.models_classes[cl]) + '_' + fname, 'wb'))

    def load(self, folder, vname):
        for file in os.listdir(folder):
            loaded_model = pickle.load(open(folder + file, 'rb'))
            self.models.append(loaded_model)
            self.models_classes.append(file.split("-")[0])
        self.vect = pickle.load(open(vname, 'rb'))
      

class SingleModel():
    def __init__(self):
        self.model = CatBoostClassifier(random_state=42)
        self.vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.99, min_df=0.05)

    def fit(self, dataset):
        dataset = dataset[['Stemmed_comment', 'sub_target']]
        features = self.vect.transform(dataset.Stemmed_comment).toarray()
        for num in range(features.shape[1]):
            dataset['lex' + str(num)] = features[:,num]
        
        labels = dataset['sub_target']
        dataset = dataset.drop(['Stemmed_comment','sub_target'],axis=1)

        self.model.fit(dataset,labels,plot=True,verbose=False)

    def predict(self, comment):
        dataset = pd.DataFrame()
        dataset.loc[0, 'Stemmed_comment'] = comment
        features = self.vect.transform(dataset.Stemmed_comment).toarray()

        for num in range(features.shape[1]):
            dataset['lex' + str(num)] = features[:,num]

        dataset = dataset.drop(['Stemmed_comment'],axis=1)
        preds = self.model.predict(dataset)
        preds_proba = self.model.predict_proba(dataset)
        
        return preds, preds_proba

    def save(self, fname, vname):
        self.model.save_model(fname)
        with open(vname, 'wb') as fin:
            pickle.dump(self.vect, fin)

    def load(self, fname, vname):
        self.model.load_model(fname)
        self.vect = pickle.load(open(vname, 'rb'))