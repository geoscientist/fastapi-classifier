import sys
import argparse
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from nlp_service.model import MetaClassifier
from nlp_service.preprocessors import fix_text, prepare_data, vectorize
from nlp_service.postprocessors import no_oper_processor, cassette_processor
from nlp_service.utils import build_final_dataset
from task_configs import get_config
from nlp_service.trainer import Trainer

warnings.filterwarnings('ignore')

logging.basicConfig(format = '[%(asctime)s] [LEVEL:%(levelname)s] %(message)s',
                        datefmt = ('%Y-%m-%d %H:%M:%S'),
                        level = logging.INFO,
                        stream = sys.stdout)

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'inference'], default='train')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-o', '--output_file')
    return parser


logging.info("starting application")
parser = createParser()
namespace = parser.parse_args()
CONFIG = get_config(namespace.config)
mode = namespace.mode


if mode == "train":
    t = Trainer(CONFIG)
    model = t.train()
elif mode == "inference":
    logging.info("check input and output files for inference")
    data = pd.read_excel(namespace.input_file, engine='openpyxl')
    result = namespace.output_file
    logging.info("starting inference on new data")
    model = MetaClassifier()
    model.load("./models/checkpoints/", "./models/vectorizer_17-01-2022.pk")
    test_data = fix_text(data, CONFIG['FEATURE_COLS'][0])
    classes_dict = {k:v for d in CONFIG['CLASSES'] for k, v in d.items()}
    data = vectorize(test_data, model.vect)
    predictions, predictions_prob, classes = model.predict(data)
    test_data = build_final_dataset(test_data, predictions, predictions_prob, classes, classes_dict)
    test_data = no_oper_processor(test_data)
    test_data = cassette_processor(test_data)
    test_data.to_excel(result)
    logging.info("inference successfully finished")
