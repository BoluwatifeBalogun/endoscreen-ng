"""Administrative module: users, screening records, datasets and models."""
from __future__ import annotations

import csv
import io
import json
import re
import threading
from collections import Counter
from datetime import timedelta
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import (Blueprint, Response, abort, current_app, flash, g, jsonify,
                   redirect, render_template, request, url_for)
from sqlalchemy import func
from werkzeug.utils import secure_filename

from . import schema as S
from .ml.train import DatasetError, clean_dataset, train_all
from .models import Dataset, MLModel, Prediction, User, db, utcnow

bp = Blueprint("admin", __name__, url_prefix="/admin")
PER_PAGE = 15


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.admin is None:
            if g.user is not None:
                abort(403, description="This area is for system administrators.")
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


# -- overview -----------------------------------------------------------------
@bp.route("/")
@admin_required
def overview():
    total = Prediction.query.count()
    by_risk = dict(db.session.query(Prediction.risk_level, func.count()).group_by(Prediction.risk_level).all())
    by_result = dict(db.session.query(Prediction.result, func.count()).group_by(Prediction.result).all())
    by_state = db.session.query(User.state, func.count(Prediction.prediction_id)) \
        .join(Prediction, Prediction.user_id == User.user_id).group_by(User.state).all()
    zones = Counter()
    for state, n in by_state:
        zones[S.STATE_TO_ZONE.get(state, "Unknown")] += n
    since = utcnow().date() - timedelta(days=13)
    recent = Prediction.query.filter(Prediction.created_at >= since).with_entities(Prediction.created_at).all()
    per_day = Counter(row.created_at.date() for row in recent)
    trend = [{"day": since + timedelta(days=i), "n": per_day.get(since + timedelta(days=i), 0)} for i in range(14)]
    return render_template(
        "admin/overview.html", total=total, by_risk=by_risk,
        by_result=[(c, by_result.get(c, 0)) for c in S.CLASSES],
        zones=[(z, zones.get(z, 0)) for z in S.NIGERIAN_STATES],
        users=User.query.count(), active_users=User.query.filter_by(is_active=True).count(),
        states_covered=len({s for s, _ in by_state}), trend=trend,
        latest=Prediction.query.order_by(Prediction.created_at.desc()).limit(6).all())


# -- users --------------------------------------------------------------------
@bp.route("/users")
@admin_required
def users():
    q = request.args.get("q", "").strip()
    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(User.name.ilike(like), User.email.ilike(like),
                                    User.facility.ilike(like), User.state.ilike(like)))
    rows = query.order_by(User.created_at.desc()).all()
    counts = dict(db.session.query(Prediction.user_id, func.count()).group_by(Prediction.user_id).all())
    return render_template("admin/users.html", rows=rows, counts=counts, q=q)


@bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def toggle_user(user_id):
    user = db.session.get(User, user_id) or abort(404)
    user.is_active = not user.is_active
    db.session.commit()
    flash(f"{user.name} has been {'reactivated' if user.is_active else 'deactivated'}.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = db.session.get(User, user_id) or abort(404)
    name = user.name
    db.session.delete(user)
    db.session.commit()
    flash(f"{name} and all of their screening records were deleted.", "success")
    return redirect(url_for("admin.users"))


# -- records --------------------------------------------------------------------
def _records_query():
    q, risk = request.args.get("q", "").strip(), request.args.get("risk", "")
    query = Prediction.query.join(User)
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(Prediction.patient_ref.ilike(like), User.name.ilike(like),
                                    User.facility.ilike(like), User.state.ilike(like)))
    if risk in ("Low", "Moderate", "High"):
        query = query.filter(Prediction.risk_level == risk)
    return query.order_by(Prediction.created_at.desc()), q, risk


@bp.route("/records")
@admin_required
def records():
    query, q, risk = _records_query()
    page = max(request.args.get("page", 1, type=int), 1)
    matched = query.count()
    rows = query.offset((page - 1) * PER_PAGE).limit(PER_PAGE).all()
    return render_template("admin/records.html", rows=rows, q=q, risk=risk, page=page, matched=matched,
                           pages=max((matched + PER_PAGE - 1) // PER_PAGE, 1))


@bp.route("/records/export.csv")
@admin_required
def export_records():
    query, _q, _risk = _records_query()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["prediction_id", "date_utc", "patient_ref", "health_worker", "facility", "state", "zone",
                     "result", "risk_level", "p_imbalance", "confidence", "model"] + S.RAW_COLUMNS)
    for p in query.all():
        inputs = p.inputs
        writer.writerow([p.prediction_id, p.created_at.isoformat(timespec="seconds"), p.patient_ref, p.user.name,
                         p.user.facility, p.user.state, S.STATE_TO_ZONE.get(p.user.state, ""), p.result,
                         p.risk_level, round(p.p_imbalance, 4), round(p.confidence, 4),
                         f"{p.model.algorithm} {p.model.version}" if p.model else ""]
                        + [inputs.get(c, "") if inputs.get(c) is not None else "" for c in S.RAW_COLUMNS])
    return Response(buffer.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=endoscreen_records.csv"})


@bp.route("/records/<int:prediction_id>/delete", methods=["POST"])
@admin_required
def delete_record(prediction_id):
    prediction = db.session.get(Prediction, prediction_id) or abort(404)
    db.session.delete(prediction)
    db.session.commit()
    flash(f"Screening record #{prediction_id} was deleted.", "success")
    return redirect(url_for("admin.records"))


# -- datasets -------------------------------------------------------------------
@bp.route("/datasets")
@admin_required
def datasets():
    rows = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    return render_template("admin/datasets.html", rows=rows)


@bp.route("/datasets/template.csv")
@admin_required
def dataset_template():
    example = {c: "" for c in S.DATASET_COLUMNS}
    example.update(age=34, sex="Female", weight_kg=72.5, height_cm=162, systolic_bp=118, diastolic_bp=76,
                   sleep_hours=7, exercise_level="Light", menstrual_pattern="Irregular", tsh=2.1,
                   glucose=92, diagnosis=S.CLASSES[3])
    example.update({c: "No" for c in S.BINARY_FIELDS})
    example.update(acne="Yes", excess_hair="Yes")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=S.DATASET_COLUMNS)
    writer.writeheader()
    writer.writerow(example)
    return Response(buffer.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=endoscreen_dataset_template.csv"})


@bp.route("/datasets/upload", methods=["POST"])
@admin_required
def upload_dataset():
    file = request.files.get("file")
    name = request.form.get("name", "").strip()
    if not file or not file.filename:
        flash("Choose a CSV file to upload.", "error")
        return redirect(url_for("admin.datasets"))
    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are accepted. Download the template to see the expected columns.", "error")
        return redirect(url_for("admin.datasets"))
    stamp = utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{stamp}_{secure_filename(file.filename)}"
    path = Path(current_app.config["DATA_DIR"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    file.save(path)
    try:
        raw = pd.read_csv(path, keep_default_na=False, na_values=["", "NA", "NaN", "nan"])
        _clean, report = clean_dataset(raw)
    except DatasetError as exc:
        path.unlink(missing_ok=True)
        flash(str(exc), "error")
        return redirect(url_for("admin.datasets"))
    except Exception:
        path.unlink(missing_ok=True)
        flash("The file could not be read as CSV. Save it as comma-separated values and try again.", "error")
        return redirect(url_for("admin.datasets"))
    dataset = Dataset(name=name or file.filename, filename=filename,
                      source=request.form.get("source", "").strip() or "Uploaded by administrator",
                      n_records=report["rows_clean"], n_features=len(S.FEATURES),
                      class_distribution=json.dumps(report["class_distribution"]),
                      uploaded_by=g.admin.admin_id, status="ready",
                      status_message=f"{report['duplicates_removed']} duplicates and "
                                     f"{report['rows_dropped_missing_essentials']} unusable rows will be removed.")
    db.session.add(dataset)
    db.session.commit()
    flash(f"Dataset uploaded: {report['rows_clean']} usable records. You can now train models on it.", "success")
    return redirect(url_for("admin.datasets"))


def _next_version() -> str:
    numbers = [int(m.group(1)) for (v,) in db.session.query(MLModel.version).distinct()
               if (m := re.fullmatch(r"v(\d+)", v))]
    on_disk = [int(m.group(1)) for p in Path(current_app.config["MODELS_DIR"]).glob("v*")
               if (m := re.fullmatch(r"v(\d+)", p.name))]
    return f"v{max(numbers + on_disk + [0]) + 1}"


def _train_job(app, dataset_id: int, version: str, fast: bool):
    from . import register_model_version
    with app.app_context():
        dataset = db.session.get(Dataset, dataset_id)
        try:
            path = Path(app.config["DATA_DIR"]) / dataset.filename
            metrics = train_all(path, Path(app.config["MODELS_DIR"]) / version, version, fast=fast,
                                log=lambda _msg: None)
            register_model_version(app, version, dataset=dataset)
            best = metrics["models"][metrics["best_model"]]
            dataset.status = "trained"
            dataset.status_message = (f"Version {version} trained. Best: {best['name']} "
                                      f"(test accuracy {best['test']['accuracy']:.1%}).")
        except Exception as exc:  # surface any failure to the admin instead of hanging
            db.session.rollback()
            dataset = db.session.get(Dataset, dataset_id)
            dataset.status = "failed"
            dataset.status_message = f"Training failed: {exc}"
        db.session.commit()


@bp.route("/datasets/<int:dataset_id>/train", methods=["POST"])
@admin_required
def train_dataset(dataset_id):
    dataset = db.session.get(Dataset, dataset_id) or abort(404)
    if Dataset.query.filter_by(status="training").count():
        flash("A training run is already in progress. Wait for it to finish.", "error")
        return redirect(url_for("admin.datasets"))
    if not (Path(current_app.config["DATA_DIR"]) / dataset.filename).exists():
        flash("The dataset file is no longer on the server. Upload it again.", "error")
        return redirect(url_for("admin.datasets"))
    version = _next_version()
    dataset.status, dataset.status_message = "training", f"Training version {version}. This takes one to three minutes."
    db.session.commit()
    app = current_app._get_current_object()
    fast = bool(request.form.get("fast"))
    if app.config.get("TRAIN_SYNC"):
        _train_job(app, dataset_id, version, fast)
    else:
        threading.Thread(target=_train_job, args=(app, dataset_id, version, fast), daemon=True).start()
    flash(f"Training of version {version} has started.", "info")
    return redirect(url_for("admin.datasets"))


@bp.route("/datasets/status")
@admin_required
def dataset_status():
    return jsonify({d.dataset_id: {"status": d.status, "message": d.status_message}
                    for d in Dataset.query.all()})


@bp.route("/datasets/<int:dataset_id>/delete", methods=["POST"])
@admin_required
def delete_dataset(dataset_id):
    dataset = db.session.get(Dataset, dataset_id) or abort(404)
    if dataset.models.count():
        flash("This dataset has trained models and is kept for audit. It cannot be deleted.", "error")
    elif dataset.status == "training":
        flash("Wait for training to finish before deleting this dataset.", "error")
    else:
        (Path(current_app.config["DATA_DIR"]) / dataset.filename).unlink(missing_ok=True)
        db.session.delete(dataset)
        db.session.commit()
        flash("Dataset deleted.", "success")
    return redirect(url_for("admin.datasets"))


# -- models ---------------------------------------------------------------------
@bp.route("/models")
@admin_required
def models():
    rows = MLModel.query.order_by(MLModel.version.desc(), MLModel.f1_score.desc()).all()
    return render_template("admin/models.html", rows=rows)


@bp.route("/models/<int:model_id>/activate", methods=["POST"])
@admin_required
def activate_model(model_id):
    model = db.session.get(MLModel, model_id) or abort(404)
    if not (Path(current_app.config["MODELS_DIR"]) / model.file_path).exists():
        flash("The file for this model is missing on the server, so it cannot be activated.", "error")
        return redirect(url_for("admin.models"))
    MLModel.query.update({MLModel.is_active: False})
    model.is_active = True
    db.session.commit()
    flash(f"{model.algorithm} {model.version} is now the active screening model.", "success")
    return redirect(url_for("admin.models"))
