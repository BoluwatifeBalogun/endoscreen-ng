"""Single source of truth for the screening feature schema.

The same definitions drive: the dataset generator, dataset validation on
upload, the training pipeline, the web form, server-side validation and the
prediction engine. Keeping them in one place guarantees that inference applies
exactly the preprocessing the model was trained with (write-up section 3.12.1).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Target classes
# --------------------------------------------------------------------------
NORMAL = "No hormonal imbalance"
CLASSES = [
    NORMAL,
    "Hypothyroidism",
    "Hyperthyroidism",
    "Polycystic ovary syndrome (PCOS)",
    "Diabetes mellitus",
    "Adrenal insufficiency",
]
CLASS_SHORT = {
    NORMAL: "No imbalance",
    "Hypothyroidism": "Hypothyroidism",
    "Hyperthyroidism": "Hyperthyroidism",
    "Polycystic ovary syndrome (PCOS)": "PCOS",
    "Diabetes mellitus": "Diabetes",
    "Adrenal insufficiency": "Adrenal insufficiency",
}
LABEL_COLUMN = "diagnosis"

# --------------------------------------------------------------------------
# Raw input fields (what a health worker enters / what a dataset CSV holds)
# --------------------------------------------------------------------------
# name: (label, unit, min, max, step, help)
CORE_NUMERIC = {
    "age": ("Age", "years", 15, 90, 1, "Adults and adolescents aged 15 and above."),
    "weight_kg": ("Weight", "kg", 30, 200, 0.1, ""),
    "height_cm": ("Height", "cm", 120, 210, 0.1, ""),
    "systolic_bp": ("Systolic blood pressure", "mmHg", 70, 240, 1, "The upper reading."),
    "diastolic_bp": ("Diastolic blood pressure", "mmHg", 40, 140, 1, "The lower reading."),
    "sleep_hours": ("Average sleep per night", "hours", 2, 14, 0.5, ""),
}

SEX_OPTIONS = ["Female", "Male"]
EXERCISE_OPTIONS = ["None", "Light", "Moderate", "Active"]
EXERCISE_HELP = {
    "None": "Mostly sitting",
    "Light": "Walks 1 to 2 days a week",
    "Moderate": "Active 3 to 4 days a week",
    "Active": "Active most days",
}
MENSTRUAL_OPTIONS = ["Regular", "Irregular", "Absent", "Not applicable"]
MENSTRUAL_HELP = {
    "Regular": "Cycles of 21 to 35 days",
    "Irregular": "Unpredictable or widely spaced",
    "Absent": "No period for 3 months or more",
    "Not applicable": "Male, pregnant or post-menopausal",
}

# name: (label, plain-language prompt)
SYMPTOMS = {
    "fatigue": ("Persistent fatigue", "Tiredness that rest does not relieve"),
    "weight_gain": ("Unexplained weight gain", "Gaining weight without eating more"),
    "weight_loss": ("Unexplained weight loss", "Losing weight without trying"),
    "acne": ("Persistent acne", "Especially jawline, chin and lower face"),
    "hair_loss": ("Hair thinning or loss", "Diffuse thinning of scalp hair"),
    "excess_hair": ("Excess facial or body hair", "Coarse hair on face, chest or abdomen"),
    "mood_swings": ("Mood swings or irritability", "Sudden, unpredictable mood changes"),
    "depression": ("Low mood or depression", "Persistent sadness or loss of interest"),
    "sleep_problems": ("Sleep problems", "Trouble falling or staying asleep"),
    "infertility": ("Difficulty conceiving", "Trying for 12 months or more"),
    "cold_intolerance": ("Feels cold easily", "Cold when others are comfortable"),
    "heat_intolerance": ("Heat intolerance or sweating", "Hot and sweaty when others are not"),
    "excessive_thirst": ("Excessive thirst", "Drinking far more than usual"),
    "frequent_urination": ("Frequent urination", "Including waking at night to pass urine"),
    "palpitations": ("Palpitations", "Fast or pounding heartbeat at rest"),
    "dizziness": ("Dizziness on standing", "Light-headed when getting up"),
    "skin_darkening": ("Skin darkening", "Neck folds, creases, gums or scars"),
    "salt_craving": ("Salt craving", "Strong urge for salty food"),
}
SYMPTOM_GROUPS = [
    ("Energy and weight", ["fatigue", "weight_gain", "weight_loss", "excessive_thirst", "frequent_urination"]),
    ("Skin and hair", ["acne", "hair_loss", "excess_hair", "skin_darkening"]),
    ("Mood and sleep", ["mood_swings", "depression", "sleep_problems"]),
    ("Temperature and circulation", ["cold_intolerance", "heat_intolerance", "palpitations", "dizziness", "salt_craving"]),
    ("Reproductive", ["infertility"]),
]
HISTORY_FLAGS = {
    "family_history": ("Family history of an endocrine disorder",
                       "Parent or sibling with diabetes, thyroid disease or PCOS"),
}

# name: (label, unit, min, max, step, ref_low, ref_high, help)
LABS = {
    "glucose": ("Fasting blood glucose", "mg/dL", 30, 700, 1, 70, 99, "Glucometer reading after an overnight fast."),
    "tsh": ("TSH", "mIU/L", 0.005, 150, 0.001, 0.4, 4.0, "Thyroid-stimulating hormone."),
    "t3": ("Total T3", "ng/dL", 20, 700, 1, 80, 200, ""),
    "t4": ("Total T4", "µg/dL", 0.5, 30, 0.1, 5.0, 12.0, ""),
    "lh": ("LH", "mIU/mL", 0.1, 120, 0.1, 1.5, 12.0, "Luteinising hormone (follicular phase for women)."),
    "fsh": ("FSH", "mIU/mL", 0.1, 150, 0.1, 1.5, 12.5, "Follicle-stimulating hormone."),
    "cortisol": ("Morning cortisol", "µg/dL", 0.2, 70, 0.1, 6.0, 23.0, "Sample taken between 7 and 9 am."),
    "insulin": ("Fasting insulin", "µIU/mL", 0.5, 300, 0.1, 2.0, 25.0, ""),
}
LAB_PANELS = [
    ("Blood sugar", ["glucose", "insulin"]),
    ("Thyroid panel", ["tsh", "t3", "t4"]),
    ("Reproductive hormones", ["lh", "fsh"]),
    ("Adrenal", ["cortisol"]),
]
MGDL_PER_MMOL = 18.016

BINARY_FIELDS = list(SYMPTOMS) + list(HISTORY_FLAGS)
RAW_COLUMNS = (
    ["age", "sex", "weight_kg", "height_cm", "systolic_bp", "diastolic_bp",
     "sleep_hours", "exercise_level", "menstrual_pattern"]
    + BINARY_FIELDS + list(LABS)
)
DATASET_COLUMNS = RAW_COLUMNS + [LABEL_COLUMN]

# --------------------------------------------------------------------------
# Model feature matrix
# --------------------------------------------------------------------------
LOG_LABS = {"tsh": "tsh_log", "insulin": "insulin_log"}
LAB_FEATURES = ["glucose", "tsh_log", "t3", "t4", "lh", "fsh", "lh_fsh_ratio", "cortisol", "insulin_log"]
MISSING_FLAGS = [f"{lab}_missing" for lab in LABS]
FEATURES = (
    ["age", "is_female", "bmi", "weight_kg", "systolic_bp", "diastolic_bp",
     "sleep_hours", "exercise_level", "menses_irregular", "menses_absent"]
    + BINARY_FIELDS + LAB_FEATURES + MISSING_FLAGS
)

FEATURE_LABELS = {
    "age": "Age", "is_female": "Sex", "bmi": "Body mass index", "weight_kg": "Weight",
    "systolic_bp": "Systolic blood pressure", "diastolic_bp": "Diastolic blood pressure",
    "sleep_hours": "Sleep duration", "exercise_level": "Exercise level",
    "menses_irregular": "Irregular menstruation", "menses_absent": "Absent menstruation",
    "tsh_log": "TSH", "insulin_log": "Fasting insulin", "lh_fsh_ratio": "LH to FSH ratio",
    "glucose": "Fasting blood glucose", "t3": "Total T3", "t4": "Total T4",
    "lh": "LH", "fsh": "FSH", "cortisol": "Morning cortisol",
}
FEATURE_LABELS.update({k: v[0] for k, v in SYMPTOMS.items()})
FEATURE_LABELS.update({k: v[0] for k, v in HISTORY_FLAGS.items()})
FEATURE_LABELS.update({f"{k}_missing": f"{v[0]} not measured" for k, v in LABS.items()})

NIGERIAN_STATES = {
    "North Central": ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau", "FCT Abuja"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
}
STATE_TO_ZONE = {s: z for z, states in NIGERIAN_STATES.items() for s in states}
CADRES = ["Doctor", "Nurse / Midwife", "Community Health Officer (CHO)",
          "Community Health Extension Worker (CHEW)", "Medical Laboratory Scientist",
          "Pharmacist", "Other health worker"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _yes(value) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"yes", "y", "1", "true", "on"} else 0.0
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return 1.0 if float(value) >= 0.5 else 0.0


def _num(value) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def bmi_of(weight_kg, height_cm) -> float:
    w, h = _num(weight_kg), _num(height_cm)
    if np.isnan(w) or np.isnan(h) or h <= 0:
        return np.nan
    return round(w / (h / 100.0) ** 2, 1)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Encode and engineer features. Used identically for training and inference.

    - label-encodes sex, exercise level and menstrual pattern
    - derives BMI and the LH:FSH ratio (feature extraction, section 2.2.11)
    - log-transforms the two heavily right-skewed labs
    - adds one 'not measured' flag per laboratory test so the model can still
      score a patient when a primary healthcare centre has no laboratory
    Missing numeric values are left as NaN; the fitted pipeline imputes them.
    """
    out = pd.DataFrame(index=raw.index)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)

    out["age"] = num("age")
    sex = raw["sex"].astype(str).str.strip().str.lower() if "sex" in raw else pd.Series("", index=raw.index)
    out["is_female"] = (sex.str[:1] == "f").astype(float)
    weight, height = num("weight_kg"), num("height_cm")
    out["bmi"] = weight / (height / 100.0) ** 2
    out["weight_kg"] = weight
    out["systolic_bp"] = num("systolic_bp")
    out["diastolic_bp"] = num("diastolic_bp")
    out["sleep_hours"] = num("sleep_hours")
    ex = raw["exercise_level"] if "exercise_level" in raw else pd.Series("None", index=raw.index)
    ex_map = {name.lower(): i for i, name in enumerate(EXERCISE_OPTIONS)}
    out["exercise_level"] = ex.astype(str).str.strip().str.lower().map(ex_map)
    mp = raw["menstrual_pattern"].astype(str).str.strip().str.lower() if "menstrual_pattern" in raw \
        else pd.Series("not applicable", index=raw.index)
    out["menses_irregular"] = ((mp == "irregular") & (out["is_female"] == 1)).astype(float)
    out["menses_absent"] = ((mp == "absent") & (out["is_female"] == 1)).astype(float)

    for col in BINARY_FIELDS:
        out[col] = raw[col].map(_yes) if col in raw else 0.0

    labs = {lab: num(lab) for lab in LABS}
    for lab in LABS:
        lo, hi = LABS[lab][2], LABS[lab][3]
        labs[lab] = labs[lab].where((labs[lab] >= lo) & (labs[lab] <= hi))
    out["glucose"] = labs["glucose"]
    out["tsh_log"] = np.log1p(labs["tsh"])
    out["t3"] = labs["t3"]
    out["t4"] = labs["t4"]
    out["lh"] = labs["lh"]
    out["fsh"] = labs["fsh"]
    out["lh_fsh_ratio"] = labs["lh"] / labs["fsh"]
    out["cortisol"] = labs["cortisol"]
    out["insulin_log"] = np.log1p(labs["insulin"])
    for lab in LABS:
        out[f"{lab}_missing"] = labs[lab].isna().astype(float)
    return out[FEATURES].astype(float)


def validate_record(data: dict) -> tuple[dict, dict]:
    """Server-side validation of one screening submission.

    Returns (clean_record, errors). `errors` maps field name to a message a
    health worker can act on. Mirrors the client-side rules (section 3.12.2).
    """
    clean, errors = {}, {}

    for name, (label, unit, lo, hi, _step, _help) in CORE_NUMERIC.items():
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = f"{label} is required."
            continue
        value = _num(raw)
        if np.isnan(value):
            errors[name] = f"{label} must be a number."
        elif not lo <= value <= hi:
            errors[name] = f"{label} must be between {lo} and {hi} {unit}."
        else:
            clean[name] = value

    def choice(name, label, options):
        raw = str(data.get(name) or "").strip()
        match = next((o for o in options if o.lower() == raw.lower()), None)
        if match is None:
            errors[name] = f"Choose an option for {label.lower()}."
        else:
            clean[name] = match

    choice("sex", "Sex", SEX_OPTIONS)
    choice("exercise_level", "Exercise level", EXERCISE_OPTIONS)
    if clean.get("sex") == "Male":
        clean["menstrual_pattern"] = "Not applicable"
    else:
        choice("menstrual_pattern", "Menstrual pattern", MENSTRUAL_OPTIONS)

    for name in BINARY_FIELDS:
        clean[name] = int(_yes(data.get(name)))
    if clean["weight_gain"] and clean["weight_loss"]:
        errors["weight_loss"] = "Choose either weight gain or weight loss, not both."
    if clean["cold_intolerance"] and clean["heat_intolerance"]:
        errors["heat_intolerance"] = "Choose either cold or heat intolerance, not both."

    if "systolic_bp" in clean and "diastolic_bp" in clean and clean["diastolic_bp"] >= clean["systolic_bp"]:
        errors["diastolic_bp"] = "Diastolic pressure must be lower than systolic pressure."

    glucose_unit = str(data.get("glucose_unit") or "mg/dL")
    for name, (label, unit, lo, hi, *_rest) in LABS.items():
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            clean[name] = None
            continue
        value = _num(raw)
        if np.isnan(value):
            errors[name] = f"{label} must be a number, or leave it blank if not measured."
            continue
        if name == "glucose" and glucose_unit.lower().startswith("mmol"):
            value = round(value * MGDL_PER_MMOL, 1)
        if not lo <= value <= hi:
            shown = f"{lo} and {hi} {unit}"
            if name == "glucose" and glucose_unit.lower().startswith("mmol"):
                shown = f"{lo / MGDL_PER_MMOL:.1f} and {hi / MGDL_PER_MMOL:.1f} mmol/L"
            errors[name] = f"{label} must be between {shown}, or left blank."
        else:
            clean[name] = value
    return clean, errors
