"""Database layer: the six tables of Table 3.1 (USER, ADMINISTRATOR, PREDICTION,
RECOMMENDATION, DATASET, ML_MODEL) with the fields of the data dictionary."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class _Password:
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class User(db.Model, _Password):
    __tablename__ = "user"
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    cadre = db.Column(db.String(80), nullable=False)
    facility = db.Column(db.String(160), nullable=False)
    state = db.Column(db.String(40), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    predictions = db.relationship("Prediction", backref="user", cascade="all, delete-orphan", lazy="dynamic")


class Administrator(db.Model, _Password):
    __tablename__ = "administrator"
    admin_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(40), default="System administrator", nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)


class Dataset(db.Model):
    __tablename__ = "dataset"
    dataset_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    source = db.Column(db.String(255), default="")
    n_records = db.Column(db.Integer, default=0)
    n_features = db.Column(db.Integer, default=0)
    class_distribution = db.Column(db.Text, default="{}")
    status = db.Column(db.String(20), default="ready")       # ready | training | trained | failed
    status_message = db.Column(db.Text, default="")
    uploaded_by = db.Column(db.Integer, db.ForeignKey("administrator.admin_id"), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    models = db.relationship("MLModel", backref="dataset", lazy="dynamic")

    @property
    def distribution(self) -> dict:
        return json.loads(self.class_distribution or "{}")


class MLModel(db.Model):
    __tablename__ = "ml_model"
    model_id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.dataset_id"), nullable=True)
    algorithm = db.Column(db.String(60), nullable=False)
    algorithm_key = db.Column(db.String(40), nullable=False)
    version = db.Column(db.String(40), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    auc_roc = db.Column(db.Float)
    is_best = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=False)
    trained_at = db.Column(db.DateTime, default=utcnow, nullable=False)


class Prediction(db.Model):
    __tablename__ = "prediction"
    prediction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=False, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey("ml_model.model_id"), nullable=True)
    patient_ref = db.Column(db.String(40), nullable=False)
    input_data = db.Column(db.Text, nullable=False)           # JSON of submitted values
    result = db.Column(db.String(80), nullable=False)         # predicted class
    risk_level = db.Column(db.String(12), nullable=False)     # Low | Moderate | High
    confidence = db.Column(db.Float, nullable=False)
    p_imbalance = db.Column(db.Float, nullable=False)
    details = db.Column(db.Text, nullable=False)              # JSON: probabilities, explanation, flags
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False, index=True)
    model = db.relationship("MLModel")
    recommendations = db.relationship("Recommendation", backref="prediction",
                                      cascade="all, delete-orphan", order_by="Recommendation.rec_id")

    @property
    def inputs(self) -> dict:
        return json.loads(self.input_data)

    @property
    def detail(self) -> dict:
        return json.loads(self.details)


class Recommendation(db.Model):
    __tablename__ = "recommendation"
    rec_id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("prediction.prediction_id"), nullable=False, index=True)
    category = db.Column(db.String(40), nullable=False)
    advice_text = db.Column(db.Text, nullable=False)
