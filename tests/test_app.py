"""Integration, system and security tests (section 3.18)."""
import io
import time

from endoscreen import schema as S
from endoscreen.ml.generate_dataset import generate
from endoscreen.models import Dataset, MLModel, Prediction, Recommendation, User, db
from tests.conftest import REGISTER, patient

HYPOTHYROID = dict(age="47", fatigue="1", weight_gain="1", cold_intolerance="1", hair_loss="1",
                   depression="1", tsh="18.5", t4="3.9", t3="78", weight_kg="82", patient_ref="LAG-7781")
DIABETES = dict(age="56", sex="Male", excessive_thirst="1", frequent_urination="1", fatigue="1",
                weight_loss="1", glucose="236", family_history="1", weight_kg="88", height_cm="172",
                systolic_bp="142", diastolic_bp="90", patient_ref="KAN-0031")
PCOS = dict(age="24", menstrual_pattern="Irregular", acne="1", excess_hair="1", weight_gain="1",
            lh="15.5", fsh="5.1", weight_kg="79", patient_ref="ENU-2210")
HEALTHY = dict(age="29", glucose="86", tsh="1.9", patient_ref="ABJ-1002")


def screen(client, **overrides):
    return client.post("/screen", data=patient(**overrides))


# ---- authentication -----------------------------------------------------------
def test_public_pages_load(client):
    for url in ["/", "/login", "/register", "/healthz"]:
        assert client.get(url).status_code == 200


def test_protected_pages_redirect_to_login(client):
    for url in ["/dashboard", "/screen", "/performance", "/admin/", "/admin/users"]:
        r = client.get(url)
        assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_register_validates_and_hashes_password(client, app):
    bad = client.post("/register", data={**REGISTER, "email": "nope", "password": "short", "confirm": "short"})
    assert bad.status_code == 422 and b"valid email" in bad.data
    assert client.post("/register", data=REGISTER).status_code == 302
    with app.app_context():
        user = User.query.filter_by(email="amina@phc.ng").one()
        assert user.password_hash != "Secret123" and user.check_password("Secret123")
    client.post("/logout")
    dup = client.post("/register", data=REGISTER)
    assert dup.status_code == 422 and b"already exists" in dup.data


def test_login_logout_and_wrong_password(user_client):
    user_client.post("/logout")
    assert user_client.get("/dashboard").status_code == 302
    assert user_client.post("/login", data={"email": "amina@phc.ng", "password": "wrong"}).status_code == 401
    ok = user_client.post("/login", data={"email": "AMINA@phc.ng", "password": "Secret123"})
    assert ok.status_code == 302 and ok.headers["Location"].endswith("/dashboard")


def test_open_redirect_is_blocked(user_client):
    user_client.post("/logout")
    r = user_client.post("/login?next=//evil.example", data={"email": "amina@phc.ng", "password": "Secret123"})
    assert r.headers["Location"].endswith("/dashboard")


def test_csrf_is_enforced_when_enabled(app):
    app.config["CSRF_ENABLED"] = True
    c = app.test_client()
    assert c.post("/login", data={"email": "a@b.ng", "password": "x"}).status_code == 400
    page = c.get("/login").data.decode()
    token = page.split('name="csrf_token" value="')[1].split('"')[0]
    assert c.post("/login", data={"email": "a@b.ng", "password": "x", "csrf_token": token}).status_code == 401


# ---- screening flow -------------------------------------------------------------
def test_screening_end_to_end(user_client, app):
    r = screen(user_client, **HYPOTHYROID)
    assert r.status_code == 302
    page = user_client.get(r.headers["Location"])
    assert page.status_code == 200
    html = page.data.decode()
    assert "High risk" in html and "Hypothyroidism" in html and "not a medical diagnosis" in html
    with app.app_context():
        p = Prediction.query.one()
        assert p.result == "Hypothyroidism" and p.risk_level == "High" and p.patient_ref == "LAG-7781"
        assert p.inputs["tsh"] == 18.5 and p.model.is_active
        cats = [rec.category for rec in p.recommendations]
        assert "Referral" in cats and "Confirmatory tests" in cats and cats[-1] == "Reminder"
        assert p.detail["drivers"][0]["label"] == "TSH" and p.detail["drivers"][0]["effect"] > 0


def test_clinical_scenarios_classify_as_expected(user_client, app):
    for overrides, expected in [(DIABETES, "Diabetes mellitus"), (PCOS, "Polycystic ovary syndrome (PCOS)")]:
        assert screen(user_client, **overrides).status_code == 302
    assert screen(user_client, **HEALTHY).status_code == 302
    with app.app_context():
        rows = Prediction.query.order_by(Prediction.prediction_id).all()
        assert [r.result for r in rows[:2]] == ["Diabetes mellitus", "Polycystic ovary syndrome (PCOS)"]
        assert rows[0].risk_level == "High" and rows[1].risk_level in ("High", "Moderate")
        assert rows[2].risk_level == "Low" and rows[2].result == S.NORMAL


def test_symptoms_only_screening_works_without_labs(user_client, app):
    r = screen(user_client, age="41", fatigue="1", weight_loss="1", dizziness="1", skin_darkening="1",
               salt_craving="1", systolic_bp="92", diastolic_bp="60", weight_kg="54", patient_ref="BEN-0090")
    assert r.status_code == 302
    with app.app_context():
        p = Prediction.query.one()
        assert p.detail["labs_provided"] == 0 and p.risk_level in ("Moderate", "High")
        assert p.detail["top_condition"]["label"] == "Adrenal insufficiency"
        assert any(rec.category == "Data quality" for rec in p.recommendations)


def test_invalid_screening_is_rejected_with_messages(user_client, app):
    r = screen(user_client, age="7", patient_ref="Mrs Ada Obi!", consent="")
    assert r.status_code == 422
    html = r.data.decode()
    assert "Age must be between 15 and 90" in html and "card or folder number" in html and "verbal consent" in html
    with app.app_context():
        assert Prediction.query.count() == 0


def test_users_cannot_see_each_others_results(user_client, app):
    loc = screen(user_client, **HEALTHY).headers["Location"]
    other = app.test_client()
    other.post("/register", data={**REGISTER, "email": "bola@phc.ng"})
    assert other.get(loc).status_code == 403
    assert other.post(loc + "/delete").status_code == 403
    assert user_client.get(loc).status_code == 200


def test_dashboard_history_filter_and_delete(user_client):
    screen(user_client, **HYPOTHYROID)
    screen(user_client, **HEALTHY)
    page = user_client.get("/dashboard").data.decode()
    assert "LAG-7781" in page and "ABJ-1002" in page
    filtered = user_client.get("/dashboard?risk=High").data.decode()
    assert "LAG-7781" in filtered and "ABJ-1002" not in filtered
    assert "ABJ-1002" in user_client.get("/dashboard?q=abj").data.decode()
    user_client.post("/result/1/delete")
    assert 'class="row-link" href="/result/1"' not in user_client.get("/dashboard").data.decode()


def test_performance_page_reports_all_three_models(user_client):
    html = user_client.get("/performance").data.decode()
    for text in ["Logistic Regression", "Random Forest", "Support Vector Machine", "AUC-ROC",
                 "Confusion matrix", "Hypothesis test", "Friedman"]:
        assert text in html
    assert user_client.get("/performance?model=random_forest").status_code == 200


def test_json_api(user_client, client):
    assert client.application.test_client().post("/api/predict", json={}).status_code == 401
    bad = user_client.post("/api/predict", json={"patient_ref": "X1", "age": 200})
    assert bad.status_code == 422 and "age" in bad.get_json()["fields"]
    payload = {k: v for k, v in patient(**DIABETES).items() if k != "consent"}
    good = user_client.post("/api/predict", json=payload)
    body = good.get_json()
    assert good.status_code == 201 and body["result"] == "Diabetes mellitus" and body["risk_level"] == "High"
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01 and body["recommendations"]


def test_prediction_response_time(user_client):
    screen(user_client, **HEALTHY)          # warm the model cache
    start = time.perf_counter()
    screen(user_client, **PCOS)
    assert time.perf_counter() - start < 2.0


# ---- administration ----------------------------------------------------------------
def test_user_cannot_open_admin(user_client):
    assert user_client.get("/admin/").status_code == 403
    assert user_client.post("/admin/users/1/delete").status_code == 403


def test_admin_pages_and_user_management(admin_client, user_client, app):
    screen(user_client, **HYPOTHYROID)
    for url in ["/admin/", "/admin/users", "/admin/records", "/admin/datasets", "/admin/models", "/performance"]:
        assert admin_client.get(url).status_code == 200, url
    assert b"South West" in admin_client.get("/admin/").data
    assert admin_client.get("/result/1").status_code == 200
    export = admin_client.get("/admin/records/export.csv")
    assert export.mimetype == "text/csv" and b"LAG-7781" in export.data

    admin_client.post("/admin/users/1/toggle")
    assert user_client.get("/dashboard").status_code == 302          # session ends at once
    assert b"deactivated" in user_client.post("/login", data={"email": "amina@phc.ng", "password": "Secret123"}).data
    admin_client.post("/admin/users/1/toggle")
    admin_client.post("/admin/users/1/delete")
    with app.app_context():
        assert User.query.count() == 0 and Prediction.query.count() == 0 and Recommendation.query.count() == 0


def test_admin_record_delete(admin_client, user_client, app):
    screen(user_client, **HEALTHY)
    admin_client.post("/admin/records/1/delete")
    with app.app_context():
        assert Prediction.query.count() == 0


def test_dataset_upload_validation(admin_client):
    r = admin_client.post("/admin/datasets/upload", data={"name": "bad", "file": (io.BytesIO(b"a,b\n1,2\n"), "bad.csv")},
                          content_type="multipart/form-data", follow_redirects=True)
    assert b"missing required columns" in r.data
    r = admin_client.post("/admin/datasets/upload", data={"name": "x", "file": (io.BytesIO(b"hi"), "notes.txt")},
                          content_type="multipart/form-data", follow_redirects=True)
    assert b"Only CSV files" in r.data
    assert admin_client.get("/admin/datasets/template.csv").data.decode().split("\n")[0].strip() == ",".join(S.DATASET_COLUMNS)


def test_upload_retrain_and_activate(admin_client, user_client, app):
    csv_bytes = generate(seed=11).sample(900, random_state=1).to_csv(index=False).encode()
    r = admin_client.post("/admin/datasets/upload", content_type="multipart/form-data", follow_redirects=True,
                          data={"name": "Pilot clinic data", "source": "test", "file": (io.BytesIO(csv_bytes), "pilot.csv")})
    assert b"Dataset uploaded" in r.data
    with app.app_context():
        ds = Dataset.query.filter_by(name="Pilot clinic data").one()
        ds_id = ds.dataset_id
    admin_client.post(f"/admin/datasets/{ds_id}/train", data={"fast": "1"})
    with app.app_context():
        ds = db.session.get(Dataset, ds_id)
        assert ds.status == "trained", ds.status_message
        v2 = MLModel.query.filter_by(version="v2").all()
        assert len(v2) == 3 and sum(m.is_best for m in v2) == 1 and not any(m.is_active for m in v2)
        target = next(m for m in v2 if m.algorithm_key == "logistic_regression").model_id
    admin_client.post(f"/admin/models/{target}/activate")
    screen(user_client, **HYPOTHYROID)
    with app.app_context():
        p = Prediction.query.one()
        assert p.model.version == "v2" and p.model.algorithm == "Logistic Regression"
        assert MLModel.query.filter_by(is_active=True).count() == 1
    assert admin_client.get("/performance?version=v2").status_code == 200
