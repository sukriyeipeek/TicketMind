from fastapi.testclient import TestClient
from ticketmind.api import create_app


def test_workbench_api_requires_attestation_and_exports_development_only(tmp_path):
    with TestClient(create_app(tmp_path / 'missing.joblib', workbench_path=tmp_path / 'local.sqlite3')) as client:
        record = {'text': 'Bu tamamen yeni bir insan test örneğidir.', 'label': 'review', 'group_id': 'api-new-scenario'}
        assert client.post('/workbench', json=record).status_code == 422
        record['author_confirmed'] = True
        assert client.post('/workbench', json=record).status_code == 201
        assert client.post('/workbench', json=record).status_code == 400
        exported = client.get('/workbench/export')
        assert 'development' in exported.text and 'human_authored' in exported.text
        assert client.get('/workbench').json()['stats']['total'] == 1


def test_unknown_model_and_external_rag_provider_rejected(tmp_path):
    with TestClient(create_app(tmp_path / 'missing.joblib')) as client:
        assert client.post('/analyze', json={'text': 'Merhaba', 'model': '../../private'}).status_code == 422
        assert client.post('/rag/draft', json={'text': 'Merhaba', 'provider': 'https://example.com'}).status_code == 422
        assert client.post('/rag/draft', json={'text': 'Astronomide karanlık madde nedir?'}).json()['status'] == 'no_evidence'
