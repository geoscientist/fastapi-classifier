import os
from datetime import date
import pandas as pd 
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from tqdm import tqdm
import fasttext
import pickle
from joblib import dump, load

class MetaClassifier:
    def __init__(self, model_arr = [], mod_cl = []):
        self.models = model_arr
        self.models_classes = mod_cl
        self.consistency = bool(len(self.models) == len(self.models_classes))


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

    def cross_validate(self, dataset, labels, verbose=True):
        for cl, model in enumerate(self.models):
            if verbose == True: print('model class:', self.models_classes[cl])
            current_target = labels.apply(lambda x: int(self.models_classes[cl] in x.split(','))).values
            skf = StratifiedKFold(n_splits=5)
            prec = []
            rec = []
            for train_index, test_index in skf.split(dataset, current_target):
                X_train_cv, X_test_cv = dataset[train_index], dataset[test_index]
                y_train_cv, y_test_cv = current_target[train_index], current_target[test_index]
                model.fit(X_train_cv,y_train_cv)
                pred = model.predict(X_test_cv)
                prec.append(precision_score(y_test_cv,pred,average=None))
                rec.append(recall_score(y_test_cv,pred,average=None))

            prec = np.mean(np.array(prec))
            rec = np.mean(np.array(rec))
            if verbose == True:
                print('Precision:',prec)
                print('Recall:',rec)

    def predict(self, test_data):
        predictions = []
        predictions_probability = []
        for model in self.models:
            pred = model.predict(test_data)
            pred_prob = model.predict_proba(test_data)
            predictions.append(pred)
            predictions_probability.append(pred_prob)
        return predictions, predictions_probability, self.models_classes

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for cl, model in enumerate(self.models):
            dump(model, open(str(folder + self.models_classes[cl]) + '-model_{}.pkl'.format(date.today().strftime("%d-%m-%Y")), 'wb'))

    def load(self, folder, vname):
        for file in os.listdir(folder):
            loaded_model = load(open(folder + file, 'rb'))
            self.models.append(loaded_model)
            self.models_classes.append(file.split("-")[0])
        self.vect = load(open(vname, 'rb'))


            
class FasttextClassifier():
    def __init__(self):
        self.model = None

    def fit(self, dataset, labels, N_epoch=20):
        #train_data = dataset.copy()
        #train_data['target'] = labels

        #print(dataset['Stemmed_comment'])

        train_data = np.hstack((dataset,np.array(['__label__' + str(x) for x in labels]).reshape(-1,1)))

        filename = 'Train_Data.txt'

        if os.path.exists(filename):
            print('File already exists, removing...')
            os.remove(filename)
            print('Done.')
        else:
            print('File does not exist!')

        np.savetxt(filename,train_data,fmt = ['%s','%s'])

        train_data.to_csv(filename, header=False, index=False, sep="\t")
        print('File created!')

        del train_data

        self.model = fasttext.train_supervised(input=filename, epoch=N_epoch)

        if os.path.exists(filename):
            print('File exists, removing...')
            os.remove(filename)
            print('Done.')
        else:
            print('File does not exist!')

    def predict(self, dataset):
        preds = []
        for comm in dataset.Stemmed_comment.values:
            preds.append(self.model.predict(comm)[0][0][9:])

        return np.array(preds)
        

class SingleClassifier():
    def __init__(self):
        self.model = CatBoostClassifier(random_state=42)
        self.vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.99, min_df=0.05)

    def fit(self, dataset):
        # We transform each complaint into a vector
        dataset = dataset[['Stemmed_comment', 'sub_target']]
        features = self.vect.transform(dataset.Stemmed_comment).toarray()
        for num in range(features.shape[1]):
            dataset['lex' + str(num)] = features[:,num]
    
        
        labels = dataset['sub_target']
        dataset = dataset.drop(['Stemmed_comment','sub_target'],axis=1)

        self.model.fit(dataset,labels,plot=True,verbose=False)

    def predict(self, dataset):
        dataset = dataset[['Stemmed_comment']]
        features = self.vect.transform(dataset.Stemmed_comment).toarray()

        for num in range(features.shape[1]):
            dataset['lex' + str(num)] = features[:,num]

        dataset = dataset.drop(['Stemmed_comment'],axis=1)
        preds = self.model.predict(dataset)
        preds_proba = self.model.predict_proba(dataset)
        
        return preds, preds_proba
    
    def cross_validate(self, dataset, verbose_cv=True):
        dataset = dataset[['Stemmed_comment', 'sub_target']]
        features = self.vect.fit_transform(dataset.Stemmed_comment).toarray()

        for num in range(features.shape[1]):
            dataset['lex' + str(num)] = features[:,num]
            
        skf = StratifiedKFold(n_splits=5)
        prec = []
        rec = []
        current_target = dataset['sub_target'].factorize()[0]
        
        dataset = dataset.drop(['Stemmed_comment','sub_target'],axis=1)
        
        for train_index, test_index in skf.split(dataset.values, current_target):
            X_train_cv, X_test_cv = dataset.loc[train_index,:], dataset.loc[test_index,:]
            y_train_cv, y_test_cv = current_target[train_index], current_target[test_index]
            self.model.fit(X_train_cv,y_train_cv,verbose=False)
            pred = self.model.predict(X_test_cv)
            prec.append(precision_score(y_test_cv,pred,average='weighted'))
            rec.append(recall_score(y_test_cv,pred,average='weighted'))

        prec = np.mean(np.array(prec))
        rec = np.mean(np.array(rec))
        if verbose_cv == True:
            print('Precision:',prec)
            print('Recall:',rec)
    
    def save(self, fname, vname):
        self.model.save_model(fname)
        with open(vname, 'wb') as fin:
            pickle.dump(self.vect, fin)

    def load(self, fname, vname):
        self.model.load_model(fname)
        self.vect = pickle.load(open(vname, 'rb'))