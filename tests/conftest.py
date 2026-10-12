import shutil
from pathlib import Path

import pytest

from endoscreen import create_app
from endoscreen.models import User, db

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def app(tmp_path):
    models = tmp_path / "models"
    shutil.copytree(ROOT / "models" / "v1", models / "v1")
    app = create_app({
        "TESTING": True, "CSRF_ENABLED": False, "TRAIN_SYNC": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "MODELS_DIR": str(models), "DATA_DIR": str(tmp_path / "data"),
    })
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


REGISTER = dict(name="Amina Okafor", email="amina@phc.ng", cadre="Nurse / Midwife",
                facility="Iwaya Primary Health Centre", state="Lagos",
                password="Secret123", confirm="Secret123", terms="1")


@pytest.fixture()
def user_client(client):
    assert client.post("/register", data=REGISTER).status_code == 302
    return client


@pytest.fixture()
def admin_client(app):
    c = app.test_client()
    r = c.post("/login", data={"email": app.config["ADMIN_EMAIL"], "password": app.config["ADMIN_PASSWORD"]})
    assert r.status_code == 302
    return c


def patient(**overrides):
    base = dict(patient_ref="PHC/0412", age="34", sex="Female", weight_kg="68", height_cm="163",
                systolic_bp="118", diastolic_bp="76", sleep_hours="7", exercise_level="Moderate",
                menstrual_pattern="Regular", consent="1")
    base.update(overrides)
    return base
