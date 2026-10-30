"""English Banking77 API. No synthetic policies or generated banking advice."""
import json
import threading
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from ticketmind.banking import BankingService, REPORT, DATA, MODELS, labels, read_csv

class AnalyzeRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text: StrictStr=Field(min_length=1,max_length=4000)
    model: Literal['word','char']='word'
    @field_validator('text')
    @classmethod
    def visible(cls,value):
        if not value.strip(): raise ValueError('Message must not be blank')
        return value

def create_app():
    app=FastAPI(title='TicketMind · Banking77',version='0.3.0')
    cache={}
    lock=threading.Lock()
    @app.get('/health')
    def health():
        return {'dataset':'banking77','language':'en','synthetic':False,'models':{k:(MODELS/(k+'.joblib')).exists() for k in ('word','char')}}
    @app.get('/labels')
    def categories(): return {'labels':labels()}
    @app.post('/analyze')
    def analyze(payload:AnalyzeRequest):
        try:
            with lock:
                if payload.model not in cache: cache[payload.model]=BankingService(payload.model)
                service=cache[payload.model]
            return service.analyze(payload.text)
        except (OSError,ValueError,AttributeError) as error:
            raise HTTPException(503,'Train Banking77 models first: python -m ticketmind.banking') from error
    @app.get('/experiments')
    def experiments():
        if not REPORT.exists(): raise HTTPException(404,'No Banking77 report yet')
        return json.loads(REPORT.read_text(encoding='utf-8'))
    @app.get('/examples')
    def examples(label:str,limit:int=10):
        if label not in labels() or not 1<=limit<=50: raise HTTPException(422,'Invalid label or limit')
        return {'source':'official_training_csv','examples':[r for r in read_csv('train.csv') if r['label']==label][:limit]}
    @app.get('/data/manifest')
    def manifest():
        path=DATA/'manifest.json'
        if not path.exists(): raise HTTPException(404,'Data preparation not completed')
        return json.loads(path.read_text(encoding='utf-8'))
    return app

app=create_app()
