"""Unit tests: validation, feature engineering, cleaning, engine rules (section 3.18)."""
import numpy as np
import pandas as pd

from endoscreen import schema as S
from endoscreen.engine import range_flags, urgent_flags
from endoscreen.ml.generate_dataset import generate
from endoscreen.ml.train import clean_dataset, derive_thresholds
from tests.conftest import patient


def test_valid_record_passes_and_male_menstrual_is_forced():
    clean, errors = S.validate_record(patient(sex="Male", menstrual_pattern="Irregular"))
    assert errors == {}
    assert clean["menstrual_pattern"] == "Not applicable"
    assert clean["tsh"] is None and clean["fatigue"] == 0


def test_required_and_range_errors():
    _, errors = S.validate_record(patient(age="", weight_kg="900", sex="", exercise_level="Sprinting"))
    assert {"age", "weight_kg", "sex", "exercise_level"} <= set(errors)
    assert "between 30 and 200" in errors["weight_kg"]


def test_blood_pressure_consistency_and_exclusive_symptoms():
    _, errors = S.validate_record(patient(systolic_bp="80", diastolic_bp="95", weight_gain="1", weight_loss="1"))
    assert "diastolic_bp" in errors and "weight_loss" in errors


def test_glucose_mmol_is_converted_to_mgdl():
    clean, errors = S.validate_record(patient(glucose="7.0", glucose_unit="mmol/L"))
    assert errors == {} and abs(clean["glucose"] - 126.1) < 0.2


def test_lab_rejects_text_and_out_of_range():
    _, errors = S.validate_record(patient(tsh="abc", cortisol="999"))
    assert "tsh" in errors and "cortisol" in errors


def test_build_features_matches_training_schema():
    clean, _ = S.validate_record(patient(lh="14", fsh="5", tsh="2.0", acne="1"))
    X = S.build_features(pd.DataFrame([clean]))
    assert list(X.columns) == S.FEATURES and X.shape == (1, len(S.FEATURES))
    row = X.iloc[0]
    assert abs(row["bmi"] - 68 / 1.63 ** 2) < 1e-6
    assert abs(row["lh_fsh_ratio"] - 2.8) < 1e-6
    assert row["is_female"] == 1 and row["acne"] == 1 and row["exercise_level"] == 2
    assert row["glucose_missing"] == 1 and row["tsh_missing"] == 0 and np.isnan(row["glucose"])
    assert abs(row["tsh_log"] - np.log1p(2.0)) < 1e-9


def test_cleaning_removes_duplicates_and_impossible_values():
    raw = generate(seed=7)
    clean, report = clean_dataset(raw)
    assert report["duplicates_removed"] >= 40
    assert report["out_of_range_values_blanked"] >= 12
    assert clean["age"].between(15, 90).all()
    assert not clean.duplicated().any()
    assert set(clean[S.LABEL_COLUMN]) == set(S.CLASSES)


def test_thresholds_are_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * .6 + rng.normal(.2, .2, 500), 0, 1)
    t = derive_thresholds(y, p)
    assert 0 < t["moderate"] < t["high"] <= .9


def test_urgent_and_range_flags():
    rec = {"glucose": 340, "systolic_bp": 84, "diastolic_bp": 52, "dizziness": 1, "cortisol": 1.8,
           "weight_kg": 50, "height_cm": 170, "tsh": 9.8}
    assert len(urgent_flags(rec)) == 3
    flags = {f["label"]: f for f in range_flags(rec)}
    assert flags["TSH"]["status"] == "High" and flags["BMI"]["status"] == "Underweight"
    assert urgent_flags({"systolic_bp": 120, "diastolic_bp": 80}) == []
