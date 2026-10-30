import hashlib
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from ticketmind.api import create_app
from ticketmind.banking import DATA, REVISION, labels, normalize, prepare, read_csv, pipeline

def test_official_data_and_label_inventory():
    assert len(labels()) == len(set(labels())) == 77
    assert len(read_csv('train.csv')) == 10003
    assert len(read_csv('test.csv')) == 3080
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert manifest['revision']==REVISION and manifest['synthetic'] is False
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest

def test_validation_groups_and_test_are_disjoint():
    fit,val,test,audit=prepare()
    keys=lambda rows:{normalize(r['text']) for r in rows}
    assert not keys(fit)&keys(val)
    assert not keys(fit)&keys(test)
    assert not keys(val)&keys(test)
    assert len(fit)+len(val)+audit['excluded_train_rows']==10003
    assert {r['label'] for r in fit}==set(labels())

def test_english_normalization():
    assert normalize(' I am  WAITING ')=='i am waiting'

def test_saved_pipeline_loads_in_another_process(tmp_path):
    import joblib
    import subprocess
    import sys
    model=pipeline('word')
    model.fit(['card card arrived','payment payment failed','card card sent','payment payment declined'],['a','b','a','b'])
    path=tmp_path/'model.joblib'
    joblib.dump(model,path)
    subprocess.run([sys.executable,'-c',"import joblib,sys; m=joblib.load(sys.argv[1]); assert m.predict(['card arrived'])[0]=='a'",str(path)],check=True,timeout=60)

@pytest.mark.parametrize('kind',['word','char'])
def test_vectorizer_fits_training_only(kind):
    model=pipeline(kind)
    model.fit(['card card arrived delivery','payment payment failed bank','card sent delivery','payment declined bank'],['a','b','a','b'])
    vocabulary=dict(model['tfidf'].vocabulary_)
    model.predict(['zzholdouttokenzz'])
    assert vocabulary==model['tfidf'].vocabulary_
    assert 'zzholdouttokenzz' not in vocabulary
    assert np.isclose(model.predict_proba(['payment failed'])[0].sum(),1)

@pytest.mark.parametrize('payload',[{}, {'text':''}, {'text':'  '}, {'text':42}, {'text':'x'*4001}, {'text':'message','model':'tinybert'}])
def test_api_rejects_invalid_inputs(payload):
    with TestClient(create_app()) as client:
        assert client.post('/analyze',json=payload).status_code==422

def test_no_synthetic_policy_or_ner_routes():
    with TestClient(create_app()) as client:
        assert client.get('/health').json()['synthetic'] is False
        assert len(client.get('/labels').json()['labels'])==77
        assert client.post('/rag/draft',json={'text':'test'}).status_code==404
        assert client.get('/evaluation/entities').status_code==404
        assert client.get('/examples',params={'label':'card_arrival'}).json()['source']=='official_training_csv'
