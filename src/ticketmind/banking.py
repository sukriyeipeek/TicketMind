"""Real Banking77 data, grouped validation and explicit official test evaluation."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from ticketmind.text import normalize_text as normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'banking77'
MODELS = ROOT / 'models' / 'banking77'
REPORT = ROOT / 'reports' / 'banking77.json'
REVISION = '57ec275d8078af65b7731c2a98be812d844a6d6b'

def labels():
    return tuple(json.loads((DATA / 'categories.json').read_text(encoding='utf-8')))

def read_csv(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as handle:
        return [{'id': name.removesuffix('.csv') + '-' + str(index), 'text': row['text'], 'label': row['category']}
                for index, row in enumerate(csv.DictReader(handle))]

def prepare():
    train, test = read_csv('train.csv'), read_csv('test.csv')
    classes = set(labels())
    assert len(classes) == 77
    assert all(row['label'] in classes and row['text'].strip() for row in train + test)
    test_keys = {normalize(row['text']) for row in test}
    conflicts = defaultdict(set)
    for row in train:
        conflicts[normalize(row['text'])].add(row['label'])
    ambiguous = {key for key, values in conflicts.items() if len(values) > 1}
    eligible = [row for row in train if normalize(row['text']) not in test_keys | ambiguous]
    groups = [normalize(row['text']) for row in eligible]
    fit_indices, validation_indices = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
        [row['text'] for row in eligible], [row['label'] for row in eligible], groups))
    fit = [eligible[index] for index in fit_indices]
    validation = [eligible[index] for index in validation_indices]
    assert not ({normalize(row['text']) for row in fit} & {normalize(row['text']) for row in validation})
    assert set(row['label'] for row in fit) == classes
    audit = {'official_train': len(train), 'official_test': len(test), 'fit': len(fit), 'validation': len(validation),
             'excluded_train_rows': len(train)-len(eligible), 'ambiguous_training_text_groups': len(ambiguous),
             'train_test_exact_overlap_groups': len(set(conflicts) & test_keys),
             'test_duplicate_rows': len(test)-len(test_keys),
             'grouping': 'NFKC + casefold + whitespace exact duplicates only; not semantic deduplication',
             'validation_seed': 42, 'validation_folds': 5}
    return fit, validation, test, audit

def pipeline(kind):
    settings = {'analyzer': 'word', 'ngram_range': (1,2), 'max_features': 50000} if kind == 'word' else {
        'analyzer': 'char_wb', 'ngram_range': (3,5), 'max_features': 70000}
    return Pipeline([('tfidf', TfidfVectorizer(preprocessor=normalize, lowercase=False, min_df=2, sublinear_tf=True, **settings)),
                     ('classifier', LogisticRegression(C=4, max_iter=300, solver='lbfgs', random_state=42))])

def evaluate(model, rows):
    truth = [row['label'] for row in rows]
    predicted = model.predict([row['text'] for row in rows])
    return {'n': len(rows), 'accuracy': float(accuracy_score(truth, predicted)),
            'macro_f1': float(f1_score(truth, predicted, labels=list(labels()), average='macro', zero_division=0)),
            'per_class': classification_report(truth, predicted, labels=list(labels()), output_dict=True, zero_division=0),
            'confusion_labels': list(labels()), 'confusion_matrix': confusion_matrix(truth, predicted, labels=list(labels())).tolist()}

def train(evaluate_test=False):
    fit, validation, test, audit = prepare()
    MODELS.mkdir(parents=True, exist_ok=True)
    results = {}
    with threadpool_limits(limits=2):
        for kind in ('word', 'char'):
            model = pipeline(kind)
            started = time.perf_counter()
            model.fit([row['text'] for row in fit], [row['label'] for row in fit])
            metadata = {'dataset': 'banking77', 'source_revision': REVISION, 'language': 'en', 'model_kind': kind,
                        'sklearn_version': sklearn.__version__, 'labels': list(labels()), 'synthetic': False,
                        'model_version': 'banking77-' + kind + '-' + REVISION[:12], 'max_iter': 300, 'C': 4}
            path = MODELS / (kind + '.joblib')
            temporary = path.with_suffix('.tmp')
            joblib.dump({'pipeline': model, 'metadata': metadata}, temporary)
            temporary.replace(path)
            results[kind] = {'fit_seconds': time.perf_counter()-started, 'validation': evaluate(model, validation), 'metadata': metadata}
        selected = max(results, key=lambda key: results[key]['validation']['macro_f1'])
        if evaluate_test:
            chosen = joblib.load(MODELS / (selected + '.joblib'))['pipeline']
            final = evaluate(chosen, test)
        else:
            final = None
    manifest = {'dataset': 'Banking77', 'source': 'https://github.com/PolyAI-LDN/task-specific-datasets',
                'revision': REVISION, 'license': 'CC-BY-4.0', 'language': 'en', 'synthetic': False,
                'sha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ('train.csv','test.csv','categories.json','LICENSE')}}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    report = {'dataset': 'banking77', 'audit': audit, 'models': results, 'selected_model': selected,
              'selection': 'validation macro F1 only; fixed settings; official test not used for fitting/selection',
              'official_test': final, 'limitations': ['Published benchmark results, not live banking performance.',
              'Uncalibrated scores; all outputs require human review.', 'No policy answers or NER span labels are provided by Banking77.']}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'audit': audit, 'validation': {k:v['validation']['macro_f1'] for k,v in results.items()},
                      'selected': selected, 'official_test': None if final is None else {'accuracy': final['accuracy'], 'macro_f1': final['macro_f1']}}))
    return report

class BankingService:
    def __init__(self, kind='word'):
        if kind not in ('word','char'):
            raise ValueError('Unknown model')
        artifact = joblib.load(MODELS / (kind + '.joblib'))
        self.model, self.metadata = artifact['pipeline'], artifact['metadata']
        if self.metadata.get('dataset') != 'banking77' or set(self.model.classes_) != set(labels()):
            raise ValueError('Incompatible Banking77 artifact')

    def analyze(self, text):
        if not isinstance(text,str) or not text.strip() or len(text)>4000:
            raise ValueError('Enter 1–4000 characters')
        with threadpool_limits(limits=2):
            vector = self.model['tfidf'].transform([text])
            scores = self.model['classifier'].predict_proba(vector)[0]
        best = int(scores.argmax())
        return {'category': str(self.model.classes_[best]) if vector.nnz else None,
                'model_score': float(scores[best]), 'category_scores': dict(zip(map(str,self.model.classes_),map(float,scores))),
                'needs_review': True, 'score_type': 'uncalibrated_model_score', 'dataset': 'banking77', 'language': 'en',
                'model_version': self.metadata['model_version']}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate only validation-selected model on official test')
    args=parser.parse_args()
    train(args.evaluate_test)

if __name__ == '__main__':
    main()
