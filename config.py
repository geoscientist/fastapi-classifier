import os

HOST = '0.0.0.0'
PORT = int(os.getenv('CS_PORT', 8083))
BASEPATH = os.getenv('CS_BASEPATH', '/api/ceo-classifier')
