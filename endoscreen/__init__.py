"""EndoScreen NG: AI-based hormonal imbalance prediction system (Flask app factory)."""
from __future__ import annotations

import json
import os
import secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask, abort, g, jsonify, render_template, request, session

from . import schema
from .engine import PredictionEngine
from .models import Administrator, Dataset, MLModel, User, db

BASE_DIR = Path(__file__).resolve().parent.parent
__version__ = "1.0.0"


def _database_uri() -> str:
    """SQLite by default; MySQL (write-up Table 3.4) or PostgreSQL via DATABASE_URL."""
    uri = os.environ.get("DATABASE_URL", "").strip()
    if not uri:
        return f"sqlite:///{BASE_DIR / 'instance' / 'endoscreen.db'}"
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql://", 1)
    if uri.startswith("mysql://"):
        uri = uri.replace("mysql://", "mysql+pymysql://", 1)
    return uri


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(BASE_DIR / "instance"))
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or "dev-only-change-me",
        SQLALCHEMY_DATABASE_URI=_database_uri(),
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        MODELS_DIR=str(BASE_DIR / "models"),
        DATA_DIR=str(BASE_DIR / "data"),
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER") or os.environ.get("FORCE_HTTPS")),
        CSRF_ENABLED=True,
        ADMIN_EMAIL=os.environ.get("ADMIN_EMAIL", "admin@endoscreen.ng"),
        ADMIN_PASSWORD=os.environ.get("ADMIN_PASSWORD", "Admin@12345"),
    )
    app.config.update(config or {})
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    app.engine = PredictionEngine(app.config["MODELS_DIR"])

    from .admin import bp as admin_bp
    from .views import bp as main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    # -- request lifecycle --------------------------------------------------
    @app.before_request
    def load_identity():
        g.user = g.admin = None
        uid, kind = session.get("uid"), session.get("kind")
        if uid and kind == "user":
            user = db.session.get(User, uid)
            if user and user.is_active:
                g.user = user
            else:
                session.clear()
        elif uid and kind == "admin":
            g.admin = db.session.get(Administrator, uid)
            if g.admin is None:
                session.clear()

    @app.before_request
    def csrf_protect():
        if not app.config["CSRF_ENABLED"] or request.method in ("GET", "HEAD", "OPTIONS"):
            return
        expected = session.get("csrf")
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not expected or not sent or not secrets.compare_digest(expected, sent):
            abort(400, description="Your session expired or the form was invalid. Reload the page and try again.")

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if g.get("user") or g.get("admin"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    @app.context_processor
    def inject():
        return {"S": schema, "csrf_token": csrf_token, "app_version": __version__,
                "active_model": get_active_model()}

    @app.template_filter("pct")
    def pct(value, digits=0):
        return "n/a" if value is None else f"{value * 100:.{digits}f}%"

    @app.template_filter("likelihood")
    def likelihood(value):
        return "<1%" if value < 0.01 else ">99%" if value > 0.99 else f"{round(value * 100)}%"

    @app.template_filter("naira_date")
    def fmt_date(value, with_time=True):
        return "" if value is None else value.strftime("%d %b %Y, %H:%M" if with_time else "%d %b %Y")

    # -- errors -----------------------------------------------------------
    def _error(code, title, default):
        def handler(err):
            message = getattr(err, "description", None) or default
            if request.path.startswith("/api/"):
                return jsonify(error=title, message=message), code
            return render_template("error.html", code=code, title=title, message=message), code
        return handler

    app.register_error_handler(400, _error(400, "Request could not be processed", "Check the form and try again."))
    app.register_error_handler(403, _error(403, "You do not have access to this page", "Sign in with an account that has access."))
    app.register_error_handler(404, _error(404, "Page not found", "The page may have been moved or deleted."))
    app.register_error_handler(413, _error(413, "File too large", "Uploads are limited to 8 MB."))
    app.register_error_handler(500, _error(500, "Something went wrong on our side", "Please try again in a moment."))

    with app.app_context():
        db.create_all()
        bootstrap(app)

    @app.cli.command("seed-demo")
    def seed_demo():
        """Create a demo health worker account for presentations."""
        if not User.query.filter_by(email="demo@endoscreen.ng").first():
            user = User(name="Amina Okafor", email="demo@endoscreen.ng", cadre=schema.CADRES[2],
                        facility="Demo Primary Health Centre", state="Lagos")
            user.set_password("Demo@12345")
            db.session.add(user)
            db.session.commit()
        print("Demo account: demo@endoscreen.ng / Demo@12345")

    return app


def get_active_model() -> MLModel | None:
    return MLModel.query.filter_by(is_active=True).order_by(MLModel.model_id.desc()).first()


def register_model_version(app, version: str, uploaded_by=None, dataset: Dataset | None = None) -> list[MLModel]:
    """Record a trained version (models/<version>/metrics.json) in DATASET and ML_MODEL."""
    metrics = app.engine.metrics(version)
    info = metrics["dataset"]
    if dataset is None:
        dataset = Dataset(name="Simulated hormonal health dataset" if version == "v1" else info["file"],
                          filename=info["file"],
                          source="Generated from published clinical reference ranges (simulated, not patient data)"
                          if version == "v1" else "Uploaded",
                          uploaded_by=uploaded_by)
        db.session.add(dataset)
    dataset.n_records = info["cleaning"]["rows_clean"]
    dataset.n_features = info["n_features"]
    dataset.class_distribution = json.dumps(info["cleaning"]["class_distribution"])
    dataset.status = "trained"
    db.session.flush()
    rows = []
    for key, m in metrics["models"].items():
        t = m["test"]
        row = MLModel(dataset_id=dataset.dataset_id, algorithm=m["name"], algorithm_key=key, version=version,
                      file_path=f"{version}/{m['file']}", accuracy=t["accuracy"], precision=t["precision"],
                      recall=t["recall"], f1_score=t["f1"], auc_roc=t["auc_roc"],
                      is_best=(key == metrics["best_model"]))
        db.session.add(row)
        rows.append(row)
    db.session.commit()
    return rows


def bootstrap(app):
    """First-run setup: default administrator and registration of shipped models."""
    if Administrator.query.count() == 0:
        admin = Administrator(name="System Administrator", email=app.config["ADMIN_EMAIL"].lower())
        admin.set_password(app.config["ADMIN_PASSWORD"])
        db.session.add(admin)
        db.session.commit()

    # A training thread cannot survive a restart; do not leave its dataset stuck.
    for stuck in Dataset.query.filter_by(status="training"):
        stuck.status, stuck.status_message = "failed", "Training was interrupted by a server restart. Start it again."
    db.session.commit()

    models_dir = Path(app.config["MODELS_DIR"])
    known = {v for (v,) in db.session.query(MLModel.version).distinct()}
    if models_dir.exists():
        for folder in sorted(p for p in models_dir.iterdir() if (p / "metrics.json").exists()):
            if folder.name not in known:
                register_model_version(app, folder.name)
    if get_active_model() is None:
        best = MLModel.query.filter_by(is_best=True).order_by(MLModel.model_id.desc()).first()
        if best:
            best.is_active = True
            db.session.commit()
