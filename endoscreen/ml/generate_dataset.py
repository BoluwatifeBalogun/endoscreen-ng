"""Generate the simulated hormonal health dataset.

WHY SIMULATED: the public datasets named in the write-up (Kaggle PCOS, UCI
thyroid, Pima diabetes) describe one disorder each, share almost no columns
and contain no adrenal cases. Stacking them would let a classifier identify
the disorder from *which columns are blank*, which is data leakage. This
generator instead draws every record from one consistent schema using
published adult reference ranges and typical presentations (Williams Textbook
of Endocrinology; ATA thyroid guidelines; Rotterdam PCOS criteria; ADA glucose
thresholds). It is NOT patient data. Replace it through Admin > Datasets when
a real, ethically cleared dataset in the same template is available.

Deliberate imperfections so that preprocessing is exercised for real:
duplicate rows, missing values, impossible entries, class overlap
(subclinical cases) and about 3% label noise.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S

CLASS_COUNTS = {
    S.NORMAL: 1100,
    "Hypothyroidism": 430,
    "Hyperthyroidism": 270,
    "Polycystic ovary syndrome (PCOS)": 480,
    "Diabetes mellitus": 520,
    "Adrenal insufficiency": 200,
}

# Baseline symptom prevalence in people without an endocrine disorder.
BASE = dict(fatigue=.18, weight_gain=.10, weight_loss=.05, acne=.10, hair_loss=.06,
            excess_hair=.03, mood_swings=.12, depression=.07, sleep_problems=.14,
            infertility=.03, cold_intolerance=.04, heat_intolerance=.04,
            excessive_thirst=.05, frequent_urination=.05, palpitations=.05,
            dizziness=.06, skin_darkening=.03, salt_craving=.03, family_history=.15)

PROFILE = {
    S.NORMAL: dict(age=(36, 14), female=.55, bmi=(24.5, 4.0), sbp=(120, 12), dbp=(78, 8),
                   sleep=(7.0, 1.0), sym={}, menses=(.10, .04)),
    "Hypothyroidism": dict(
        age=(45, 14), female=.75, bmi=(27.5, 4.5), sbp=(124, 13), dbp=(83, 9), sleep=(7.8, 1.2),
        sym=dict(fatigue=.78, weight_gain=.62, cold_intolerance=.66, hair_loss=.42, depression=.40,
                 mood_swings=.28, sleep_problems=.28, infertility=.12, family_history=.32),
        menses=(.36, .08)),
    "Hyperthyroidism": dict(
        age=(38, 13), female=.72, bmi=(21.5, 3.5), sbp=(132, 14), dbp=(76, 9), sleep=(5.8, 1.2),
        sym=dict(fatigue=.50, weight_loss=.70, heat_intolerance=.68, palpitations=.72, mood_swings=.52,
                 sleep_problems=.56, hair_loss=.24, family_history=.30),
        menses=(.30, .08)),
    "Polycystic ovary syndrome (PCOS)": dict(
        age=(26, 6), female=1.0, bmi=(29.0, 5.0), sbp=(121, 11), dbp=(79, 8), sleep=(6.8, 1.1),
        sym=dict(acne=.62, excess_hair=.65, weight_gain=.58, hair_loss=.32, mood_swings=.40,
                 infertility=.40, skin_darkening=.34, fatigue=.35, family_history=.36),
        menses=(.68, .20)),
    "Diabetes mellitus": dict(
        age=(52, 13), female=.48, bmi=(29.5, 5.5), sbp=(134, 15), dbp=(84, 9), sleep=(6.5, 1.2),
        sym=dict(excessive_thirst=.70, frequent_urination=.72, fatigue=.62, weight_loss=.36,
                 weight_gain=.20, skin_darkening=.20, family_history=.50),
        menses=(.14, .05)),
    "Adrenal insufficiency": dict(
        age=(42, 14), female=.55, bmi=(21.0, 3.2), sbp=(99, 10), dbp=(63, 8), sleep=(7.2, 1.3),
        sym=dict(fatigue=.85, weight_loss=.64, dizziness=.70, skin_darkening=.60, salt_craving=.58,
                 depression=.34, mood_swings=.30, family_history=.18),
        menses=(.26, .10)),
}


def _lognormal(rng, median, sigma, n):
    return rng.lognormal(np.log(median), sigma, n)


def _labs(rng, label, n, female):
    lab = {
        "glucose": rng.normal(88, 9, n),
        "tsh": _lognormal(rng, 1.7, .45, n),
        "t3": rng.normal(125, 22, n),
        "t4": rng.normal(8.1, 1.4, n),
        "lh": np.where(female, rng.normal(6.2, 2.6, n), rng.normal(4.6, 1.8, n)),
        "fsh": np.where(female, rng.normal(6.6, 2.4, n), rng.normal(5.2, 2.0, n)),
        "cortisol": rng.normal(13.5, 3.6, n),
        "insulin": _lognormal(rng, 8.0, .45, n),
    }
    if label == "Hypothyroidism":
        overt = rng.random(n) > .22            # the rest are subclinical and overlap with normal
        lab["tsh"] = np.where(overt, _lognormal(rng, 14, .65, n) + 4, rng.uniform(4.1, 8.5, n))
        lab["t4"] = np.where(overt, rng.normal(4.2, 1.0, n), rng.normal(6.6, 1.1, n))
        lab["t3"] = np.where(overt, rng.normal(82, 20, n), rng.normal(108, 20, n))
    elif label == "Hyperthyroidism":
        overt = rng.random(n) > .2
        lab["tsh"] = np.where(overt, _lognormal(rng, .05, .8, n), rng.uniform(.1, .45, n))
        lab["t4"] = np.where(overt, rng.normal(15.2, 2.6, n), rng.normal(11.2, 1.3, n))
        lab["t3"] = np.where(overt, rng.normal(245, 55, n), rng.normal(185, 25, n))
    elif label == "Polycystic ovary syndrome (PCOS)":
        lab["lh"] = rng.normal(13.0, 4.6, n)
        lab["fsh"] = rng.normal(5.6, 1.8, n)
        lab["insulin"] = _lognormal(rng, 17, .5, n)
        lab["glucose"] = rng.normal(96, 11, n)
    elif label == "Diabetes mellitus":
        lab["glucose"] = _lognormal(rng, 62, .55, n) + 108
        lab["insulin"] = _lognormal(rng, 14, .7, n)
    elif label == "Adrenal insufficiency":
        lab["cortisol"] = rng.normal(3.3, 1.7, n)
        lab["glucose"] = rng.normal(77, 10, n)
    for name, values in lab.items():
        lo, hi = S.LABS[name][2], S.LABS[name][3]
        lab[name] = np.clip(values, lo, hi)
    return lab


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for label, n in CLASS_COUNTS.items():
        p = PROFILE[label]
        age = np.clip(rng.normal(*p["age"], n), 15, 88).round()
        if label.startswith("Polycystic"):
            age = np.clip(age, 15, 45)
        female = rng.random(n) < p["female"]
        bmi = np.clip(rng.normal(*p["bmi"], n), 15.5, 48)
        height = np.where(female, rng.normal(161, 6.5, n), rng.normal(172, 7, n)).clip(140, 198)
        weight = bmi * (height / 100) ** 2
        df = pd.DataFrame({
            "age": age,
            "sex": np.where(female, "Female", "Male"),
            "weight_kg": weight.round(1),
            "height_cm": height.round(1),
            "systolic_bp": np.clip(rng.normal(*p["sbp"], n) + (age - 40) * .25, 75, 220).round(),
            "diastolic_bp": np.clip(rng.normal(*p["dbp"], n), 45, 125).round(),
            "sleep_hours": (np.clip(rng.normal(*p["sleep"], n), 3, 12) * 2).round() / 2,
            "exercise_level": rng.choice(
                S.EXERCISE_OPTIONS, n,
                p=[.26, .36, .26, .12] if label == S.NORMAL else [.38, .36, .19, .07]),
        })
        irregular, absent = p["menses"]
        roll = rng.random(n)
        pattern = np.where(roll < absent, "Absent", np.where(roll < absent + irregular, "Irregular", "Regular"))
        post_meno = age >= rng.normal(50, 3, n)
        df["menstrual_pattern"] = np.where(female & ~post_meno, pattern, "Not applicable")

        for name in S.BINARY_FIELDS:
            prob = p["sym"].get(name, BASE[name])
            df[name] = np.where(rng.random(n) < prob, "Yes", "No")
        df.loc[(df.weight_gain == "Yes") & (df.weight_loss == "Yes"), "weight_gain"] = "No"
        df.loc[(df.cold_intolerance == "Yes") & (df.heat_intolerance == "Yes"), "cold_intolerance"] = "No"
        df.loc[df.sex == "Male", "excess_hair"] = np.where(rng.random((df.sex == "Male").sum()) < .01, "Yes", "No")

        for name, values in _labs(rng, label, n, female).items():
            df[name] = values.round(3 if name == "tsh" else 1)
        df[S.LABEL_COLUMN] = label
        frames.append(df)

    data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(data)

    # Laboratory availability mirrors primary care: a glucometer is common,
    # hormone assays are not. Missingness is independent of the diagnosis.
    panels = {("glucose",): .25, ("insulin",): .65, ("tsh", "t3", "t4"): .45,
              ("lh", "fsh"): .60, ("cortisol",): .62}
    for cols, p_missing in panels.items():
        mask = rng.random(n) < p_missing
        data.loc[mask, list(cols)] = np.nan
    data.loc[rng.random(n) < .08, "t3"] = np.nan

    # ~3% label noise: misdiagnosis exists in any clinical record set.
    noisy = rng.random(n) < .03
    data.loc[noisy, S.LABEL_COLUMN] = rng.choice(S.CLASSES, noisy.sum())
    data.loc[(data.sex == "Male") & data[S.LABEL_COLUMN].str.startswith("Polycystic"), S.LABEL_COLUMN] = S.NORMAL

    # Data-quality defects for the cleaning step to catch.
    data.loc[rng.choice(n, 45, replace=False), "sleep_hours"] = np.nan
    data.loc[rng.choice(n, 20, replace=False), "diastolic_bp"] = np.nan
    bad = rng.choice(n, 18, replace=False)
    data.loc[bad[:6], "age"] = [0, 2, 150, 199, 1, 300]
    data.loc[bad[6:12], "height_cm"] = [16.5, 17.0, 1.6, 15.8, 1.7, 16.2]   # metres / unit slips
    data.loc[bad[12:], "systolic_bp"] = [12, 13, 1200, 11, 1300, 14]
    dupes = data.sample(40, random_state=seed + 1)
    data = pd.concat([data, dupes], ignore_index=True).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    return data[S.DATASET_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/hormonal_health_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {args.out}")
    print(df[S.LABEL_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
