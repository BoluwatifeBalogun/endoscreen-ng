"""General health recommendations attached to each prediction.

Scope (write-up 1.8): advice is limited to referral, confirmatory tests,
lifestyle and follow-up. The system never prescribes treatment or medication.
"""
from __future__ import annotations

from . import schema as S

CONFIRMATORY = {
    "Hypothyroidism": "Request a thyroid function test (TSH and free T4) to confirm an underactive thyroid.",
    "Hyperthyroidism": "Request a thyroid function test (TSH, free T4 and T3) to confirm an overactive thyroid.",
    "Polycystic ovary syndrome (PCOS)": "Request a pelvic ultrasound and a hormone profile (LH, FSH, testosterone); "
                                        "also check fasting glucose because PCOS is linked to insulin resistance.",
    "Diabetes mellitus": "Repeat fasting blood glucose on a separate day or request HbA1c to confirm; "
                         "check urine for glucose and ketones if available.",
    "Adrenal insufficiency": "Request an early-morning (7 to 9 am) serum cortisol and electrolytes (sodium, potassium).",
}
REFERRAL = {
    "Hypothyroidism": "Refer to a physician or endocrinologist for assessment of thyroid function.",
    "Hyperthyroidism": "Refer to a physician or endocrinologist; palpitations with weight loss should be seen promptly.",
    "Polycystic ovary syndrome (PCOS)": "Refer to a gynaecologist or endocrinologist for PCOS assessment.",
    "Diabetes mellitus": "Refer to a physician or the nearest diabetes clinic for confirmation and care planning.",
    "Adrenal insufficiency": "Refer to a secondary or tertiary hospital promptly; adrenal insufficiency can worsen "
                             "quickly during illness or stress.",
}
LIFESTYLE = {
    "Hypothyroidism": "Encourage a balanced diet that includes iodised salt, and regular gentle activity.",
    "Hyperthyroidism": "Advise the patient to limit caffeine and to rest well while awaiting review.",
    "Polycystic ovary syndrome (PCOS)": "Encourage regular physical activity and a diet lower in refined sugar; "
                                        "modest weight reduction often improves cycle regularity.",
    "Diabetes mellitus": "Advise reducing sugary drinks and refined carbohydrates, regular walking, and "
                         "keeping well hydrated.",
    "Adrenal insufficiency": "Advise the patient not to skip meals, to keep hydrated and to report any vomiting, "
                             "fever or severe weakness immediately.",
}
EDUCATION = {
    "Hypothyroidism": "Explain that tiredness, weight gain and feeling cold can come from a slow thyroid and that a "
                      "simple blood test can confirm it.",
    "Hyperthyroidism": "Explain that weight loss, heat intolerance and a fast heartbeat can come from an overactive "
                       "thyroid and that a blood test can confirm it.",
    "Polycystic ovary syndrome (PCOS)": "Explain that irregular periods with acne or excess hair can point to PCOS, "
                                        "a common and manageable hormonal condition.",
    "Diabetes mellitus": "Explain that thirst, frequent urination and tiredness can be signs of raised blood sugar, "
                         "which needs a confirmatory test.",
    "Adrenal insufficiency": "Explain that persistent weakness, dizziness on standing and darker skin can be signs "
                             "that the adrenal glands are under-producing hormones.",
}


def build(result: dict) -> list[tuple[str, str]]:
    """Return (category, advice_text) pairs for a prediction result."""
    recs: list[tuple[str, str]] = []
    risk = result["risk_level"]
    condition = result["top_condition"]["label"]

    for flag in result.get("urgent_flags", []):
        recs.append(("Urgent", flag))

    if risk == "Low":
        recs.append(("Follow-up", "No pattern of hormonal imbalance was detected. Reassure the patient and "
                                  "re-screen if symptoms persist beyond four weeks or new symptoms appear."))
        recs.append(("Lifestyle", "Encourage a balanced diet, regular physical activity, 7 to 8 hours of sleep "
                                  "and routine health checks."))
        if result["top_condition"]["probability"] >= 0.15:
            recs.append(("Watch", f"The closest pattern was {S.CLASS_SHORT[condition].lower()}, at low probability. "
                                  "Note it in the patient record for comparison at the next visit."))
    else:
        if risk == "High":
            recs.append(("Referral", REFERRAL[condition]))
        else:
            recs.append(("Referral", "Discuss with the supervising clinician. " + REFERRAL[condition]))
        recs.append(("Confirmatory tests", CONFIRMATORY[condition]))
        second = result["probabilities"][1] if result["probabilities"][0]["label"] == condition \
            else result["probabilities"][0]
        if second["label"] not in (condition, S.NORMAL) and second["probability"] >= 0.20:
            recs.append(("Confirmatory tests",
                         f"{S.CLASS_SHORT[second['label']]} is also possible. " + CONFIRMATORY[second["label"]]))
        recs.append(("Patient education", EDUCATION[condition]))
        recs.append(("Lifestyle", LIFESTYLE[condition]))
        recs.append(("Follow-up", "Record this screening, give the patient a referral note and review the "
                                  "confirmatory results within two weeks."))

    if result.get("labs_provided", 0) == 0:
        recs.append(("Data quality", "This screening used symptoms and vital signs only. Adding even a fasting "
                                     "glucose or TSH result will make the prediction more reliable."))
    recs.append(("Reminder", "This is a screening aid to support your clinical judgement. It is not a diagnosis "
                             "and does not replace laboratory investigation or specialist review."))
    return recs
