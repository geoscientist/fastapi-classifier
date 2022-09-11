import sys
import logging
from fastapi import FastAPI
import pandas as pd
import re
import uvicorn
from v1 import utils
from nlp_service.model import MetaClassifier
from nlp_service.preprocessors import fix_text, vectorize
from nlp_service.postprocessors import no_oper_processor, cassette_processor
from nlp_service.utils import build_final_dataset
from config import HOST, PORT, BASEPATH
from task_configs import get_config
from pydantic import BaseModel

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from starlette.staticfiles import StaticFiles


logging.basicConfig(format = '[%(asctime)s] [LEVEL:%(levelname)s] %(message)s',
                        datefmt = ('%Y-%m-%d %H:%M:%S'),
                        level = logging.INFO,
                        stream = sys.stdout)



logging.info("STARTING SERVICE")
app = FastAPI(title="CEO classifier", version="0.2", openapi_url=f"{BASEPATH}/openapi.json", docs_url=None)
app.mount(f"{BASEPATH}/static", StaticFiles(directory="static"), name="static")
models_path = "./models/"

logging.info("Loading models...")
try:
    v1_model = utils.load_models(models_path)
    v1_host_model, _ = utils.load_single_models(models_path)
    logging.info("v1 models successfully loaded")
    model = MetaClassifier()
    model.load("./models/checkpoints/", "./models/vectorizer_17-01-2022.pk")
    logging.info("v2 models successfully loaded")
except: logging.fatal("PROBLEM WITH LOADING MODEL CHECKPOINTS!!!")
CONFIG = get_config("./task_configs/ceo.yml")
classes_dict = {k:v for d in CONFIG['CLASSES'] for k, v in d.items()}


@app.get(f"{BASEPATH}/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url=f"{BASEPATH}/static/swagger-ui-bundle.js",
        swagger_css_url=f"{BASEPATH}/static/swagger-ui.css",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

class Data(BaseModel):
    comment: str


#route for 1st version classifier
@app.post(f'{BASEPATH}/v1/predict')
async def predict(data: Data):
    comment = utils.fix_text(data.comment)
    preds, probs, classes = v1_model.predict(comment)
    logging.info(classes, probs)
    resp = utils.postprocessing(comment, preds, probs, classes, models_path)
    return resp

#route for 2nd version classifier
@app.post(f'{BASEPATH}/v2/predict')
async def predict(data: Data):
    df = pd.DataFrame(columns=['Комментарий'])
    df.loc[0, 'Комментарий'] = data.comment
    df = fix_text(df, 'Комментарий')
    vect_data = vectorize(df, model.vect)
    predictions, predictions_prob, classes = model.predict(vect_data)
    logging.info(classes, predictions_prob)
    df = build_final_dataset(df, predictions, predictions_prob, classes, classes_dict)
    df = no_oper_processor(df)
    df = cassette_processor(df)
    classes = []
    class_probas = []
    for cl in df.loc[0, 'sub_target'].split(","):
        if cl == "empty":
            classes.append(cl)
            class_probas.append(df.loc[0, 'empty_prediction_proba'])
        elif cl in ['no_oper_BNA', 'no_oper_CH', 'no_oper_payment']:
            classes.append(cl)
            class_probas.append(df.loc[0, 'no_oper_prediction_proba'])
        elif df.loc[0, cl+'_prediction_proba'] > classes_dict[cl]:
            classes.append(cl)
            class_probas.append(classes_dict[cl])
    resp = dict(zip(classes, class_probas))
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
