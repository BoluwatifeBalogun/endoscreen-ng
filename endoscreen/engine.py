"""Prediction engine: wraps the trained model behind one function.

validate -> encode/scale exactly as in training -> predict -> risk band ->
explanation -> safety flags. The web app and the JSON API both call
`PredictionEngine.predict` (the API layer of section 3.13.2).
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import schema as S


class ModelNotReady(RuntimeError):
    pass


class PredictionEngine:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self._cache: dict[str, tuple] = {}
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------
    def metrics(self, version: str) -> dict:
        path = self.models_dir / version / "metrics.json"
        if not path.exists():
            raise ModelNotReady(f"No metrics found for model version {version}.")
        return json.loads(path.read_text())

    def load(self, version: str, algorithm_key: str):
        key = f"{version}/{algorithm_key}"
        with self._lock:
            if key not in self._cache:
                path = self.models_dir / version / f"{algorithm_key}.joblib"
                if not path.exists():
                    raise ModelNotReady(f"Model file {path.name} for {version} is missing. "
                                        "Run: python -m endoscreen.ml.train")
                self._cache[key] = (joblib.load(path), self.metrics(version))
            return self._cache[key]

    # -- prediction --------------------------------------------------------
    def predict(self, record: dict, version: str, algorithm_key: str) -> dict:
        pipeline, metrics = self.load(version, algorithm_key)
        X = S.build_features(pd.DataFrame([record]))
        proba = pipeline.predict_proba(X)[0]
        classes = list(pipeline.classes_)
        probs = sorted(({"label": c, "short": S.CLASS_SHORT[c], "probability": float(p)}
                        for c, p in zip(classes, proba)), key=lambda d: -d["probability"])
        p_normal = float(proba[classes.index(S.NORMAL)])
        p_imbalance = 1.0 - p_normal
        thresholds = metrics["models"][algorithm_key].get("risk_thresholds", {"moderate": .35, "high": .65})
        risk = "High" if p_imbalance >= thresholds["high"] else \
               "Moderate" if p_imbalance >= thresholds["moderate"] else "Low"
        top_condition = next(p for p in probs if p["label"] != S.NORMAL)
        labs_provided = sum(1 for lab in S.LABS if record.get(lab) not in (None, ""))

        result = {
            "predicted_class": probs[0]["label"],
            "confidence": probs[0]["probability"],
            "p_imbalance": p_imbalance,
            "risk_level": risk,
            "thresholds": thresholds,
            "top_condition": top_condition,
            "probabilities": probs,
            "bmi": S.bmi_of(record.get("weight_kg"), record.get("height_cm")),
            "labs_provided": labs_provided,
            "drivers": self._explain(pipeline, X, record, metrics["reference"], classes, top_condition["label"]),
            "range_flags": range_flags(record),
            "urgent_flags": urgent_flags(record),
            "model": {"algorithm": metrics["models"][algorithm_key]["name"], "version": metrics["version"]},
        }
        result["headline"] = headline(result)
        result["summary"] = summary(result)
        return result

    @staticmethod
    def _explain(pipeline, X, record, reference, classes, condition) -> list[dict]:
        """Model-agnostic occlusion: swap one finding at a time for the value
        typical of a person with no imbalance and measure how far the risk moves."""
        groups = []   # (label, detail, {feature: replacement})

        def ref(feature):
            value = reference.get(feature)
            return np.nan if value is None else value

        bmi = S.bmi_of(record.get("weight_kg"), record.get("height_cm"))
        groups.append(("Body mass index", f"{bmi} kg/m²", {"bmi": ref("bmi"), "weight_kg": ref("weight_kg")}))
        groups.append(("Blood pressure", f"{record['systolic_bp']:.0f}/{record['diastolic_bp']:.0f} mmHg",
                       {"systolic_bp": ref("systolic_bp"), "diastolic_bp": ref("diastolic_bp")}))
        groups.append(("Age", f"{record['age']:.0f} years", {"age": ref("age")}))
        groups.append(("Sleep duration", f"{record['sleep_hours']:g} hours a night", {"sleep_hours": ref("sleep_hours")}))
        groups.append(("Exercise level", str(record["exercise_level"]), {"exercise_level": ref("exercise_level")}))
        if record.get("sex") == "Female" and record.get("menstrual_pattern") in ("Irregular", "Absent"):
            groups.append((f"{record['menstrual_pattern']} menstruation", "Reported",
                           {"menses_irregular": 0.0, "menses_absent": 0.0}))
        for name in S.BINARY_FIELDS:
            if record.get(name):
                label = S.SYMPTOMS.get(name, S.HISTORY_FLAGS.get(name))[0]
                groups.append((label, "Reported", {name: 0.0}))
        lab_feature = {"glucose": "glucose", "tsh": "tsh_log", "t3": "t3", "t4": "t4",
                       "cortisol": "cortisol", "insulin": "insulin_log"}
        for lab, feature in lab_feature.items():
            if record.get(lab) is not None:
                label, unit, *_r, lo, hi, _h = S.LABS[lab]
                groups.append((label, f"{record[lab]:g} {unit} (reference {lo:g} to {hi:g})", {feature: ref(feature)}))
        lh, fsh = record.get("lh"), record.get("fsh")
        if lh is not None or fsh is not None:
            detail = " and ".join(f"{n.upper()} {v:g}" for n, v in (("lh", lh), ("fsh", fsh)) if v is not None)
            if lh is not None and fsh:
                detail += f", ratio {lh / fsh:.1f}"
            repl = {}
            if lh is not None:
                repl["lh"] = ref("lh")
            if fsh is not None:
                repl["fsh"] = ref("fsh")
            if lh is not None and fsh is not None:
                repl["lh_fsh_ratio"] = ref("lh_fsh_ratio")
            groups.append(("LH and FSH", detail + " mIU/mL", repl))

        variants = pd.concat([X] * (len(groups) + 1), ignore_index=True)
        for i, (_label, _detail, repl) in enumerate(groups, start=1):
            for feature, value in repl.items():
                variants.loc[i, feature] = value
        proba = pipeline.predict_proba(variants)
        normal_idx, cond_idx = classes.index(S.NORMAL), classes.index(condition)
        # Effects are measured in log-odds, not probability points: a result of
        # 99.9% barely moves in probability when one finding is removed, yet that
        # finding may be carrying most of the evidence.
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        risk = logit(1 - proba[:, normal_idx])
        cond = logit(proba[:, cond_idx])
        drivers = []
        for i, (label, detail, _repl) in enumerate(groups, start=1):
            drivers.append({"label": label, "detail": detail,
                            "effect": float(risk[0] - risk[i]),
                            "condition_effect": float(cond[0] - cond[i])})
        peak = max((abs(d["effect"]) for d in drivers), default=0.0)
        drivers = [d for d in drivers if abs(d["effect"]) >= 0.25 and abs(d["effect"]) >= 0.08 * peak]
        drivers.sort(key=lambda d: -abs(d["effect"]))
        scale = max(peak, 1.5)        # small effects get short bars even when nothing is larger
        for d in drivers:
            size = abs(d["effect"])
            d["share"] = round(size / scale, 3)
            d["strength"] = "Strong" if size >= 1.5 else "Moderate" if size >= 0.6 else "Slight"
        return drivers[:8]


# -- plain-language helpers -----------------------------------------------------
def headline(result: dict) -> str:
    return f"{result['risk_level']} risk of hormonal imbalance"


def summary(result: dict) -> str:
    cond = result["top_condition"]
    raising = [_lower(d["label"]) for d in result["drivers"] if d["effect"] >= 0.6][:3]
    text = f"The model puts the likelihood of a hormonal imbalance at {display_pct(result['p_imbalance'])}"
    if result["risk_level"] == "Low":
        text += ", which falls in the low-risk band."
        if raising:
            text += " It did note " + _join(raising) + ", but not strongly enough to flag."
        return text
    text += f". The pattern is closest to {cond['label']} ({display_pct(cond['probability'])})."
    if raising:
        text += " The findings that weighed most were " + _join(raising) + "."
    return text


def _lower(label: str) -> str:
    """Lower-case the first letter unless the label starts with an acronym (TSH, LH)."""
    return label if label[:2].isupper() else label[0].lower() + label[1:]


def display_pct(p: float) -> str:
    """Never show 0% or 100%: a screening model is not that certain."""
    return "under 1%" if p < 0.01 else "over 99%" if p > 0.99 else f"{round(p * 100)}%"


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def range_flags(record: dict) -> list[dict]:
    flags = []
    bmi = S.bmi_of(record.get("weight_kg"), record.get("height_cm"))
    if not math.isnan(bmi):
        band = "Underweight" if bmi < 18.5 else "Healthy weight" if bmi < 25 else \
               "Overweight" if bmi < 30 else "Obese"
        flags.append({"label": "BMI", "value": f"{bmi}", "unit": "kg/m²", "status": band,
                      "level": "ok" if band == "Healthy weight" else "warn"})
    sbp, dbp = record.get("systolic_bp"), record.get("diastolic_bp")
    if sbp and dbp:
        status, level = ("High", "warn") if sbp >= 140 or dbp >= 90 else \
                        ("Low", "warn") if sbp < 90 or dbp < 60 else ("Normal", "ok")
        flags.append({"label": "Blood pressure", "value": f"{sbp:.0f}/{dbp:.0f}", "unit": "mmHg",
                      "status": status, "level": level})
    for lab, (label, unit, _lo, _hi, _step, ref_lo, ref_hi, _help) in S.LABS.items():
        value = record.get(lab)
        if value is None:
            continue
        status, level = ("Low", "warn") if value < ref_lo else ("High", "warn") if value > ref_hi else ("Normal", "ok")
        flags.append({"label": label, "value": f"{value:g}", "unit": unit, "status": status, "level": level,
                      "reference": f"{ref_lo:g} to {ref_hi:g}"})
    return flags


def urgent_flags(record: dict) -> list[str]:
    """Rule-based safety net, independent of the model, for values that need
    same-day attention whatever the predicted class."""
    flags = []
    glucose, cortisol = record.get("glucose"), record.get("cortisol")
    sbp, dbp = record.get("systolic_bp") or 0, record.get("diastolic_bp") or 0
    if glucose is not None and glucose >= 300:
        flags.append(f"Fasting glucose of {glucose:g} mg/dL is very high. Arrange same-day medical review.")
    if glucose is not None and glucose < 54:
        flags.append(f"Glucose of {glucose:g} mg/dL is dangerously low. The patient needs immediate medical attention.")
    if sbp >= 180 or dbp >= 120:
        flags.append(f"Blood pressure of {sbp:.0f}/{dbp:.0f} mmHg is severely raised. Arrange same-day medical review.")
    if 0 < sbp < 90 and record.get("dizziness"):
        flags.append(f"Low blood pressure ({sbp:.0f}/{dbp:.0f} mmHg) with dizziness. Arrange same-day medical review.")
    if cortisol is not None and cortisol < 3:
        flags.append(f"Morning cortisol of {cortisol:g} µg/dL is very low. Refer urgently to a hospital.")
    return flags
