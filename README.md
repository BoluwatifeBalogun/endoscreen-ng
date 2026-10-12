# EndoScreen NG

AI-based hormonal imbalance prediction system for primary healthcare in Nigeria.
Implementation of the seminar write-up "AI-Based Hormonal Imbalance Prediction
System Using Machine Learning" (Chapters 1 to 3).

A health worker enters a patient's measurements, symptoms and any available lab
values; the system returns a risk band, the most likely disorder, the reasons
behind the prediction and referral advice. It is a screening aid, not a diagnosis.

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python wsgi.py                                       # http://localhost:5000
```

The trained models ship in `models/v1`, so the app works immediately.

| Account | Email | Password |
| --- | --- | --- |
| Administrator (created on first run) | admin@endoscreen.ng | Admin@12345 |
| Demo health worker (`flask --app wsgi seed-demo`) | demo@endoscreen.ng | Demo@12345 |

Change the administrator password through `ADMIN_PASSWORD` before hosting.

## What is inside

| Path | Purpose |
| --- | --- |
| `endoscreen/schema.py` | Feature schema, validation and feature engineering shared by training and inference |
| `endoscreen/ml/generate_dataset.py` | Simulated dataset generator (see the note below) |
| `endoscreen/ml/train.py` | Cleaning, preprocessing, LR vs RF vs SVM, metrics, hypothesis test, export |
| `endoscreen/engine.py` | Prediction engine: risk band, explanation, safety flags |
| `endoscreen/recommendations.py` | Referral, confirmatory test and lifestyle advice |
| `endoscreen/models.py` | USER, ADMINISTRATOR, PREDICTION, RECOMMENDATION, DATASET, ML_MODEL |
| `endoscreen/views.py`, `admin.py` | Web pages, JSON API, admin module |
| `tests/` | Unit, integration, system, security and response-time tests |
| `docs/IMPLEMENTATION_PLAN.md` | Algorithm, flow, architecture, work plan, mapping to the write-up |

## Retraining

```bash
python -m endoscreen.ml.generate_dataset            # rebuilds data/hormonal_health_dataset.csv
python -m endoscreen.ml.train                       # about one minute; writes models/v1
```

Or sign in as administrator: Datasets > upload a CSV in the template format >
Train > Models > Activate.

## Tests

```bash
python -m pytest -q        # 29 tests
```

## The dataset is simulated

`data/hormonal_health_dataset.csv` (3,040 raw rows, 2,988 after cleaning) is
generated from published adult reference ranges and typical presentations. It
is not patient data. The public datasets named in the write-up cover one
disorder each with almost no shared columns, so stacking them would let a model
identify the disorder from which columns are blank. Every figure on the model
performance page therefore describes the simulated data and must be
re-established on real, ethically cleared records before any clinical use. The
admin upload and retrain feature exists for that purpose.

## Hosting on Render (free plan)

1. Push this folder to GitHub and create a Render Web Service from it (`render.yaml` is included).
2. Set `ADMIN_PASSWORD`, and set `DATABASE_URL` to a free external MySQL or PostgreSQL
   database. Without it the app uses SQLite on Render's disk, which is wiped on every
   redeploy and restart, so accounts and records would disappear.
3. Models retrained on the server are also stored on that disk. To keep a retrained
   version, train locally and commit the new `models/vN` folder.

## JSON API

`POST /api/predict` with the session cookie and an `X-CSRF-Token` header (value from
the `csrf-token` meta tag). Body: the same fields as the screening form. Returns the
class, risk level, probabilities, summary and recommendations.
