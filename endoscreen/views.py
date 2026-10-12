"""Public pages, authentication, screening, results, history and the JSON API."""
from __future__ import annotations

import json
import re
from datetime import timedelta
from functools import wraps

from flask import (Blueprint, abort, current_app, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)

from . import recommendations
from . import schema as S
from .engine import ModelNotReady
from .models import (Administrator, MLModel, Prediction, Recommendation, User,
                     db, utcnow)

bp = Blueprint("main", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
PER_PAGE = 10


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith("/api/"):
                return jsonify(error="Authentication required"), 401
            if g.admin is not None:
                return redirect(url_for("admin.overview"))
            flash("Sign in to continue.", "info")
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def _safe_next(target: str | None) -> str | None:
    return target if target and target.startswith("/") and not target.startswith("//") else None


def _active_model() -> MLModel:
    from . import get_active_model
    model = get_active_model()
    if model is None:
        raise ModelNotReady("No trained model is active yet. Ask the administrator to train and activate one.")
    return model


# -- public ------------------------------------------------------------------
@bp.route("/")
def landing():
    stats = None
    try:
        model = _active_model()
        metrics = current_app.engine.metrics(model.version)
        m = metrics["models"][model.algorithm_key]
        stats = {"accuracy": m["test"]["accuracy"], "auc": m["test"]["auc_roc"],
                 "sensitivity": m["test"]["screening"]["sensitivity"],
                 "records": metrics["dataset"]["cleaning"]["rows_clean"], "algorithm": m["name"]}
    except ModelNotReady:
        pass
    return render_template("landing.html", stats=stats)


@bp.route("/healthz")
def healthz():
    from . import get_active_model
    return jsonify(status="ok", model_ready=get_active_model() is not None)


# -- authentication ------------------------------------------------------------
def password_problems(password: str) -> str | None:
    if len(password) < 8:
        return "Use at least 8 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Include at least one letter and one number."
    return None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("main.dashboard"))
    errors, form = {}, {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in ("name", "email", "cadre", "facility", "state")}
        form["email"] = form["email"].lower()
        password, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        if len(form["name"]) < 3:
            errors["name"] = "Enter your full name."
        if not EMAIL_RE.match(form["email"]):
            errors["email"] = "Enter a valid email address, for example name@clinic.ng."
        elif User.query.filter_by(email=form["email"]).first() or \
                Administrator.query.filter_by(email=form["email"]).first():
            errors["email"] = "An account with this email already exists. Sign in instead."
        if form["cadre"] not in S.CADRES:
            errors["cadre"] = "Choose your professional cadre."
        if len(form["facility"]) < 3:
            errors["facility"] = "Enter the name of your health facility."
        if form["state"] not in S.STATE_TO_ZONE:
            errors["state"] = "Choose the state where your facility is located."
        problem = password_problems(password)
        if problem:
            errors["password"] = problem
        elif password != confirm:
            errors["confirm"] = "The two passwords do not match."
        if not request.form.get("terms"):
            errors["terms"] = "Confirm that you will use the system as a screening aid only."
        if not errors:
            user = User(**form)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            session.clear()
            session.permanent = True
            session.update(uid=user.user_id, kind="user")
            flash(f"Welcome, {user.name.split()[0]}. Your account is ready.", "success")
            return redirect(url_for("main.dashboard"))
    return render_template("register.html", errors=errors, form=form), (422 if errors else 200)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("main.dashboard"))
    if g.admin:
        return redirect(url_for("admin.overview"))
    error, email = None, ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        account = User.query.filter_by(email=email).first()
        kind = "user"
        if account is None:
            account, kind = Administrator.query.filter_by(email=email).first(), "admin"
        if account is None or not account.check_password(password):
            error = "Email or password is incorrect."
        elif kind == "user" and not account.is_active:
            error = "This account has been deactivated. Contact the administrator."
        else:
            session.clear()
            session.permanent = True
            session.update(uid=account.user_id if kind == "user" else account.admin_id, kind=kind)
            if kind == "admin":
                return redirect(url_for("admin.overview"))
            return redirect(_safe_next(request.args.get("next")) or url_for("main.dashboard"))
    return render_template("login.html", error=error, email=email), (401 if error else 200)


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("main.landing"))


# -- dashboard and history ----------------------------------------------------------
@bp.route("/dashboard")
@login_required
def dashboard():
    base = Prediction.query.filter_by(user_id=g.user.user_id)
    total = base.count()
    counts = {level: base.filter_by(risk_level=level).count() for level in ("Low", "Moderate", "High")}
    week = base.filter(Prediction.created_at >= utcnow() - timedelta(days=7)).count()

    q = request.args.get("q", "").strip()
    risk = request.args.get("risk", "")
    query = base
    if q:
        query = query.filter(Prediction.patient_ref.ilike(f"%{q}%"))
    if risk in counts:
        query = query.filter_by(risk_level=risk)
    page = max(request.args.get("page", 1, type=int), 1)
    matched = query.count()
    rows = query.order_by(Prediction.created_at.desc()).offset((page - 1) * PER_PAGE).limit(PER_PAGE).all()
    pages = max((matched + PER_PAGE - 1) // PER_PAGE, 1)
    return render_template("dashboard.html", total=total, counts=counts, week=week, rows=rows,
                           q=q, risk=risk, page=page, pages=pages, matched=matched)


# -- screening -----------------------------------------------------------------
def run_screening(user: User, patient_ref: str, record: dict) -> Prediction:
    """Predict, attach recommendations and persist. Shared by the form and the API."""
    model = _active_model()
    result = current_app.engine.predict(record, model.version, model.algorithm_key)
    prediction = Prediction(
        user_id=user.user_id, model_id=model.model_id, patient_ref=patient_ref,
        input_data=json.dumps(record), result=result["predicted_class"], risk_level=result["risk_level"],
        confidence=result["confidence"], p_imbalance=result["p_imbalance"], details=json.dumps(result))
    for category, text in recommendations.build(result):
        prediction.recommendations.append(Recommendation(category=category, advice_text=text))
    db.session.add(prediction)
    db.session.commit()
    return prediction


def _patient_ref(raw: str, errors: dict) -> str:
    ref = (raw or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9/\-]{1,19}", ref):
        errors["patient_ref"] = ("Enter the patient's card or folder number (2 to 20 letters, digits, "
                                 "hyphens or slashes). Do not enter the patient's name.")
    return ref


@bp.route("/screen", methods=["GET", "POST"])
@login_required
def screen():
    errors, form = {}, {}
    if request.method == "POST":
        form = request.form.to_dict()
        record, errors = S.validate_record(form)
        ref = _patient_ref(form.get("patient_ref"), errors)
        if not form.get("consent"):
            errors["consent"] = "Confirm that the patient has given verbal consent for this screening."
        if not errors:
            try:
                prediction = run_screening(g.user, ref, record)
            except ModelNotReady as exc:
                flash(str(exc), "error")
            else:
                return redirect(url_for("main.result", prediction_id=prediction.prediction_id))
    labels = {**{k: v[0] for k, v in S.CORE_NUMERIC.items()}, **{k: v[0] for k, v in S.LABS.items()},
              **{k: v[0] for k, v in S.SYMPTOMS.items()},
              "patient_ref": "Patient reference", "sex": "Sex", "exercise_level": "Exercise level",
              "menstrual_pattern": "Menstrual pattern", "consent": "Consent"}
    return render_template("screen.html", errors=errors, form=form, labels=labels), (422 if errors else 200)


@bp.route("/result/<int:prediction_id>")
def result(prediction_id):
    prediction = db.session.get(Prediction, prediction_id) or abort(404)
    if g.admin is None and (g.user is None or prediction.user_id != g.user.user_id):
        if g.user is None:
            return redirect(url_for("main.login", next=request.path))
        abort(403, description="This screening belongs to another health worker.")
    return render_template("result.html", p=prediction, r=prediction.detail, inputs=prediction.inputs)


@bp.route("/result/<int:prediction_id>/delete", methods=["POST"])
@login_required
def delete_result(prediction_id):
    prediction = db.session.get(Prediction, prediction_id) or abort(404)
    if prediction.user_id != g.user.user_id:
        abort(403)
    db.session.delete(prediction)
    db.session.commit()
    flash(f"Screening for {prediction.patient_ref} was deleted.", "success")
    return redirect(url_for("main.dashboard"))


# -- model transparency ----------------------------------------------------------
@bp.route("/performance")
def performance():
    if g.user is None and g.admin is None:
        return redirect(url_for("main.login", next=request.path))
    try:
        active = _active_model()
        version = request.args.get("version") or active.version
        metrics = current_app.engine.metrics(version)
    except ModelNotReady as exc:
        return render_template("error.html", code=503, title="No trained model yet", message=str(exc)), 503
    selected = request.args.get("model")
    if selected not in metrics["models"]:
        selected = active.algorithm_key if active.version == version else metrics["best_model"]
    versions = [v for (v,) in db.session.query(MLModel.version).distinct().order_by(MLModel.version)]
    return render_template("performance.html", m=metrics, selected=selected, active=active, versions=versions)


# -- JSON API ------------------------------------------------------------------
@bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object of screening fields."), 400
    record, errors = S.validate_record(payload)
    ref = _patient_ref(payload.get("patient_ref"), errors)
    if errors:
        return jsonify(error="Validation failed", fields=errors), 422
    try:
        prediction = run_screening(g.user, ref, record)
    except ModelNotReady as exc:
        return jsonify(error=str(exc)), 503
    r = prediction.detail
    return jsonify(
        prediction_id=prediction.prediction_id, result=prediction.result, risk_level=prediction.risk_level,
        confidence=round(prediction.confidence, 4), p_imbalance=round(prediction.p_imbalance, 4),
        probabilities={p["label"]: round(p["probability"], 4) for p in r["probabilities"]},
        summary=r["summary"], urgent_flags=r["urgent_flags"],
        recommendations=[{"category": rec.category, "advice": rec.advice_text} for rec in prediction.recommendations],
        model=r["model"], disclaimer="Screening aid only. Not a medical diagnosis."), 201
