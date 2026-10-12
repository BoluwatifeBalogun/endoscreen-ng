# EndoScreen NG: implementation plan

Implementation of "AI-Based Hormonal Imbalance Prediction System Using Machine
Learning". Case study: Nigeria as a whole (36 states and the FCT). Theme: green and white.

## 1. Requirements taken from the write-up

| Write-up | Requirement | Where it is implemented |
| --- | --- | --- |
| Obj. 1, 3.7 | Dataset with demographics, symptoms, biochemical markers | `ml/generate_dataset.py`, `data/` |
| Obj. 2, 2.2.10 | Cleaning, missing values, encoding, scaling, feature selection | `ml/train.py` (`clean_dataset`, pipeline), `schema.build_features` |
| Obj. 3 | Train and compare Logistic Regression, Random Forest, SVM | `ml/train.py` |
| Obj. 4, 3.12.2 | Simple web interface, instant prediction | `views.py`, `templates/screen.html`, `result.html` |
| Obj. 5, 3.17 | Accuracy, precision, recall, F1, AUC-ROC, confusion matrix | `ml/train.py`, Model performance page |
| 1.6 | H0 vs H1 on algorithm performance | Friedman test in `ml/train.py`, shown on the performance page |
| 1.8 | Registration, login, data entry, prediction, recommendations, admin module | `views.py`, `admin.py` |
| 1.8 | No treatment or prescription | `recommendations.py` limits advice to referral, tests, lifestyle |
| 3.5, 3.16 | 80:20 split, k-fold cross-validation | Stratified 80:20; 5-fold tuning; 10-fold comparison |
| 3.7.5 | Anonymised data, consent, restricted access, screening label | Card number only, consent tick, per-user access, disclaimers |
| 3.12.1 | Layered client-server architecture, prediction engine, API layer | Section 3 below, `engine.py`, `/api/predict` |
| 3.12.2 | Input validation on client and server | `static/js/app.js`, `schema.validate_record` |
| 3.12.2 | Result with confidence, explanation, recommendations; history dashboard | `result.html`, `dashboard.html` |
| 3.12.2 | Colour cues green/amber/red, responsive | Risk badges with colour, icon and text; mobile layouts |
| Table 3.1, 3.2 | Six tables and data dictionary | `models.py` |
| Table 3.4 | Python, Flask, scikit-learn, pandas, NumPy, MySQL, HTML/CSS/JS | `requirements.txt`; MySQL through `DATABASE_URL` |
| 3.18 | Unit, integration, system, performance, security testing | `tests/` (29 tests) |
| 2.3 (TAM) | Usefulness and ease of use | Four-step guided form, plain-language output |

## 2. Algorithm

### 2.1 Inputs (37 raw fields)

- Demographics and vitals: age, sex, weight, height, systolic and diastolic blood pressure.
- Lifestyle and history: sleep hours, exercise level, family history of endocrine disease.
- Symptoms (yes/no): fatigue, weight gain, weight loss, acne, hair loss, excess hair,
  mood swings, depression, sleep problems, difficulty conceiving, cold intolerance,
  heat intolerance, excessive thirst, frequent urination, palpitations, dizziness on
  standing, skin darkening, salt craving; menstrual pattern for female patients.
- Optional labs: fasting glucose, TSH, T3, T4, LH, FSH, morning cortisol, fasting insulin.

### 2.2 Target classes

No hormonal imbalance, Hypothyroidism, Hyperthyroidism, Polycystic ovary syndrome,
Diabetes mellitus, Adrenal insufficiency (the five disorders named in the summary).

### 2.3 Training procedure (CRISP-DM)

```
1  Load CSV
2  CLEAN     drop exact duplicates; drop rows with unknown labels;
             blank values outside plausible ranges; drop rows with no usable
             age, weight or height
3  ENGINEER  BMI = weight / height^2;  LH:FSH ratio;
             label-encode sex, exercise, menstrual pattern;
             log(1 + x) for TSH and insulin (right-skewed);
             one "not measured" flag per lab test            -> 46 features
4  SPLIT     stratified 80:20 train/test (seed 42)
5  For each algorithm in {Logistic Regression, Random Forest, SVM (RBF)}:
     pipeline = median imputation -> z-score scaling -> SelectKBest(ANOVA F) -> classifier
     tune hyper-parameters by 5-fold grid search on the training split (macro F1)
     collect 10-fold out-of-fold predictions on the training split
     derive risk cut-offs from those predictions (section 2.5)
     evaluate once on the held-out test split
     evaluate again with every lab value removed
6  SELECT    best = highest mean 10-fold macro F1 (test set not used for selection)
7  TEST H0   Friedman test across the three algorithms on the 10 matched folds,
             per metric; pairwise Wilcoxon signed-rank with Holm correction
8  EXPLAIN   permutation importance of the best model on the test split
9  EXPORT    three .joblib pipelines + metrics.json + healthy reference vector
```

Class imbalance is handled with class weights rather than SMOTE, so no synthetic
minority rows are added on top of the dataset.

### 2.4 Prediction procedure

```
1  Validate the submission (required fields, ranges, consistency)
2  Build the same 46 features as in training
3  p = pipeline.predict_proba(x)                     six class probabilities
4  risk = 1 - p[No hormonal imbalance]
5  band = High if risk >= t_high, Moderate if risk >= t_moderate, else Low
6  top condition = most probable disorder
7  Explanation: for each finding, replace it with the typical healthy value,
   re-score, and record the change in log-odds of risk
8  Safety net (rules, independent of the model): very high or very low glucose,
   severe hypertension, hypotension with dizziness, very low cortisol
9  Build recommendations; save PREDICTION and RECOMMENDATION rows; show result
```

### 2.5 Risk bands

Cut-offs are set per model from out-of-fold training predictions:
`t_moderate` is the highest cut-off that still flags at least 95% of true cases
(capped at 0.50) and `t_high` the lowest at which at least 95% of healthy records
fall below. For the shipped SVM these are 0.50 and 0.60.

## 3. Architecture and flow

```
Browser (HTML, CSS, JS)
   |  HTTPS, session cookie, CSRF token
Flask web application  (views.py, admin.py)       presentation + validation
   |
Prediction engine      (engine.py)                preprocessing + model + explanation
   |                        |
ML model (joblib)      Database (SQLAlchemy: SQLite / MySQL / PostgreSQL)
```

User flow: Register or sign in -> Dashboard -> New screening (1 Patient, 2 Symptoms,
3 Lifestyle, 4 Lab values) -> validation -> prediction -> Result (risk gauge,
likelihood by condition, why this result, reference ranges, next steps, print) ->
saved to history.

Administrator flow: Sign in -> Overview (usage by zone, trend) -> Users (deactivate,
delete) -> Screening records (search, export, delete) -> Datasets (upload, validate,
train) -> Models (compare, activate).

## 4. Database

| Table | Key fields |
| --- | --- |
| user | user_id, name, email, password_hash, cadre, facility, state, is_active, created_at |
| administrator | admin_id, name, email, password_hash, role |
| prediction | prediction_id, user_id, model_id, patient_ref, input_data (JSON), result, risk_level, confidence, p_imbalance, details (JSON), created_at |
| recommendation | rec_id, prediction_id, category, advice_text |
| dataset | dataset_id, name, filename, source, n_records, n_features, class_distribution, status, uploaded_by |
| ml_model | model_id, dataset_id, algorithm, version, file_path, accuracy, precision, recall, f1_score, auc_roc, is_best, is_active |

Relationships: user 1..* prediction; prediction 1..* recommendation;
dataset 1..* ml_model; ml_model 1..* prediction; administrator 1..* dataset.

## 5. Tools

Python 3.13, Flask 3, Flask-SQLAlchemy, scikit-learn, pandas, NumPy, SciPy, joblib,
HTML5, CSS3, vanilla JavaScript, pytest, gunicorn, Git.

## 6. Front-end design sources

- ui-ux-pro-max skill: healthcare guidance (accessible contrast, visible focus, 44px
  touch targets, risk never shown by colour alone, reduced-motion support, error
  summary plus inline errors), the Figtree and Noto Sans pairing, pre-delivery checklist.
- 21st.dev component patterns, re-implemented in plain CSS and JS: spotlight cards,
  border beam, aurora background, number ticker, shimmer button, text reveal.
- Framer templates and Awwwards health sites: floating glass pill navigation, bento
  feature grid, large tight headlines, soft shadows, one colour family plus white.

## 7. Work plan (iterative increments)

| Increment | Output |
| --- | --- |
| 1 | Feature schema, dataset, training pipeline, metrics, hypothesis test |
| 2 | Database models, registration, login, roles, CSRF |
| 3 | Screening form, validation, prediction engine, recommendations |
| 4 | Result page, history dashboard, model performance page |
| 5 | Admin module: users, records, datasets, retraining, model activation |
| 6 | Testing (29 automated tests, browser walkthrough), packaging, hosting files |

## 8. Deviations from the write-up, and why

1. **Dataset is simulated, not downloaded from Kaggle/UCI.** The public PCOS, thyroid
   and diabetes datasets share almost no columns and none contains adrenal cases.
   Merged, the pattern of blank columns would reveal the class (data leakage). The
   generator uses one schema for all classes, with realistic overlap, missing labs,
   duplicates, entry errors and about 3% label noise. Results must be re-established
   on real data; the admin upload and retrain feature supports this.
2. **Six output classes instead of a single yes/no.** The summary asks the system to
   "classify hormonal disorders" and the research gap criticises single-disorder tools.
   The binary low/moderate/high output of section 3.12.2 is derived from the same model.
3. **SQLite by default, MySQL by configuration.** The schema is identical; set
   `DATABASE_URL=mysql://...` to use MySQL as in Table 3.4.
4. **Section 2.5 mentions XGBoost and neural networks; the objectives and hypotheses
   name LR, RF and SVM.** The build follows the objectives. The write-up should be made
   consistent on this point.
