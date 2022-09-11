import os
import sys
import warnings
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from .model import MetaClassifier
from .preprocessors import fix_text, prepare_data
from .postprocessors import no_oper_processor
from .utils import build_final_dataset, classification_report_multiclass

logging.basicConfig(format = '[%(asctime)s] [LEVEL:%(levelname)s] %(message)s',
                        datefmt = ('%Y-%m-%d %H:%M:%S'),
                        level = logging.INFO,
                        stream = sys.stdout)

class Trainer():
    def __init__(self, CONFIG):
        self.task = CONFIG['TASK']
        self.train_data = pd.read_excel(CONFIG['TRAIN_DATA'], engine='openpyxl')
        self.test_data = pd.read_excel(CONFIG['TEST_DATA'], engine='openpyxl')
        self.classes = CONFIG['CLASSES']
        self.preprocessors = CONFIG['PREPROCESSORS']
        self.postprocessors = CONFIG['POSTPROCESSORS']
        self.feature_cols = CONFIG['FEATURE_COLS']
        self.target_col = CONFIG['TARGET_COL']

    def train(self):
        logging.info("preprocessing started")
        self.train_data = fix_text(self.train_data, self.feature_cols[0])
        self.test_data = fix_text(self.test_data, self.feature_cols[0])
        logging.info("preprecessing finished")
        self.X, self.y = prepare_data(self.train_data, self.test_data, cv=True, save_idf=True)
        classes_dict = {k:v for d in self.classes for k, v in d.items()}
        logging.info("initialize model")
        model = MetaClassifier(model_arr = [RandomForestClassifier(random_state=42) for i in range(len(classes_dict))], 
        mod_cl = list(classes_dict.keys()))
        logging.info(model.consistency)
        #model.cross_validate(self.X, self.y) # TODO add cross-validation
        # TODO add model_selecting and hyperparameter tunung
        model.fit(self.X, self.y)
        model.save("./models/checkpoints/")
        return model
        