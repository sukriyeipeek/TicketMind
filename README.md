# TicketMind · Banking77

English banking intent classification using the original **PolyAI Banking77** dataset. The active application uses **no AI-generated training/evaluation messages or fictional bank policies**. It predicts 77 intents and displays original training examples.

## Dataset and attribution

Banking77: 10,003 official training messages and 3,080 official test messages, 77 intents. Source: [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets), revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Licensed under **CC BY 4.0**; retained in `data/banking77/LICENSE`. Dataset authors: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić, *Efficient Intent Detection with Dual Sentence Encoders* (2020). Source CSVs are unchanged; hashes and source attribution are in `data/banking77/manifest.json`.

## Results

| Model | Validation macro F1 |
|---|---:|
| Word TF-IDF + Logistic Regression | 0.8841 |
| Character TF-IDF + Logistic Regression | 0.8931 |

The validation-selected character model scored **0.8962 macro F1 / 0.8964 accuracy** on the official test set. These are published-benchmark results, not a guarantee of live banking performance.

Training uses 7,995 messages; validation has 2,001. Seven official training rows with normalized exact matches in the official test were excluded from fitting/validation. One duplicate remains in the unchanged official test. NFKC/casefold/whitespace groups remain in one split. This is exact-text deduplication, not semantic duplicate detection. Validation uses the first fold of fixed five-fold stratified group splitting (seed42). Both model settings were fixed before test evaluation; only the validation-selected model was evaluated on test. The report contains per-class metrics and confusion matrices: `reports/banking77.json`.

## Install and run

Python 3.12, Windows x64 tested:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m ticketmind.banking
```

The source CSVs and license are included. To explicitly reproduce the official test measurement, add `--evaluate-test`. The default command trains and compares using validation only. Generated model files stay local, excluded from Git. Previously trained Turkish TinyBERT artifacts are not loaded by this version.

In two terminals:

```powershell
python -m uvicorn ticketmind.api:app --host 127.0.0.1 --port 8000
python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

Set `TICKETMIND_API_URL=http://127.0.0.1:18010/analyze` if the API uses port18010; Streamlit's port can similarly be changed. The current Banking77 desktop demo uses 18010/18510.

API: `/health`, `/labels`, `/analyze`, `/examples`, `/experiments`, `/data/manifest`. Analysis accepts `{"text":"English message", "model":"word"}` or `char`. Input validation caps messages at4000characters. Scores are uncalibrated; human review is always required. A 77-class model may assign an unrelated message to a banking intent. This is not a reliable out-of-domain detector and executes no financial actions.

## Scope of the migration

The earlier synthetic Turkish corpus, fictional FAQ documents, synthetic NER evaluation and generated-policy response routes were removed from this branch. Historical versions remain in Git history. Banking77 supplies **intent labels**, not support answers, policy documents or entity spans. Inventing replacements would violate the real-data-only requirement. Consequently the active UI has English classification, original training examples, evaluation and provenance tabs. There is no RAG policy generator or learned NER/transformer in this Banking77 release; they need appropriately sourced data and a separately evaluated English model.

## Reading and tests

Read `data/banking77/manifest.json`, then `src/ticketmind/banking.py` (`prepare`, `pipeline`, `train`, `BankingService`), `api.py`, and `demo/app.py`.

```powershell
python -m pytest -q
python -m pip check
```

Tests cover official counts/hashes, all77 labels, disjoint train/validation/test normalized text, training-only vectorization, strict API inputs and removal of synthetic-policy routes. Small unit-test strings are code fixtures, not added to the business training or evaluation corpus.
