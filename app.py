#!/usr/bin/env python3
"""Local chromium colorimetry training and prediction web app.

The app intentionally uses Python's standard-library HTTP server so the tool can
run in a lightweight local environment. The image/model stack is OpenCV,
scikit-learn, NumPy, and joblib. Ultralytics YOLO is used when available; the
OpenCV detector remains as a deterministic fallback and for pseudo-label
generation before a YOLO model exists.
"""

from __future__ import annotations

import argparse
import base64
import cgi
import hashlib
import json
import math
import mimetypes
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index.html"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "chromium_model.joblib"
YOLO_MODEL_PATH = MODEL_DIR / "spot_yolo.pt"
ARCHIVE_PATH = MODEL_DIR / "training_archive.joblib"
TRAINING_IMAGE_DIR = MODEL_DIR / "training_images"
WORK_DIR = APP_DIR / "runs"
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK_DIR / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(WORK_DIR / "matplotlib_config"))

MAX_REJECT_THUMBS = 80
MAX_PROCESS_THUMBS = 48
ADMIN_PASSWORD = "ywfxsy"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def json_response(handler: SimpleHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def error_response(handler: SimpleHTTPRequestHandler, message: str, status: int = 400) -> None:
    json_response(handler, {"ok": False, "error": message}, status=status)


def read_image_bytes(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片文件")
    return img


def encode_overlay(img_bgr: np.ndarray, bbox: tuple[float, float, float, float] | None, status: str, max_width: int = 900) -> str:
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        radius = max(4, int(round(min(x2 - x1, y2 - y1) / 2)))
        color = (60, 210, 60) if status == "ok" else (40, 160, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
        cv2.circle(canvas, (cx, cy), int(radius * 0.7), (255, 180, 20), 4)
        cv2.circle(canvas, (cx, cy), int(radius * 1.2), (70, 180, 255), 3)
    cv2.putText(canvas, status, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 6, cv2.LINE_AA)
    cv2.putText(canvas, status, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    if w > max_width:
        scale = max_width / w
        canvas = cv2.resize(canvas, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = float(max(0, min(width - 2, x1)))
    y1 = float(max(0, min(height - 2, y1)))
    x2 = float(max(x1 + 2, min(width - 1, x2)))
    y2 = float(max(y1 + 2, min(height - 1, y2)))
    return x1, y1, x2, y2


def candidate_score(lab_small: np.ndarray, cx: int, cy: int, radius: int) -> float:
    h, w = lab_small.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = dist <= radius * 0.68
    annulus = (dist >= radius * 1.10) & (dist <= radius * 1.42)
    if inner.sum() < 150 or annulus.sum() < 150:
        return -1.0
    inner_mean = lab_small[inner].mean(axis=0)
    ann_mean = lab_small[annulus].mean(axis=0)
    contrast = float(np.linalg.norm(inner_mean - ann_mean))
    default_cx = w * 0.52
    default_cy = h * 0.535
    default_r = min(w, h) * 0.32
    center_penalty = 0.010 * math.hypot(cx - default_cx, cy - default_cy)
    radius_penalty = 0.012 * abs(radius - default_r)
    return contrast - center_penalty - radius_penalty


def auto_detect_spot(img_bgr: np.ndarray) -> dict[str, Any]:
    """Find the center spot by contrast scoring near the expected geometry."""

    h, w = img_bgr.shape[:2]
    default_cx = int(round(w * 0.52))
    default_cy = int(round(h * 0.535))
    default_r = int(round(min(w, h) * 0.32))

    small_w = 360
    scale = small_w / w
    small_h = int(round(h * scale))
    small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    cx_vals = np.linspace(small_w * 0.47, small_w * 0.57, 6).astype(int)
    cy_vals = np.linspace(small_h * 0.48, small_h * 0.59, 6).astype(int)
    r_vals = np.linspace(min(small_w, small_h) * 0.28, min(small_w, small_h) * 0.37, 6).astype(int)
    best: tuple[float, int, int, int] | None = None
    for cx in cx_vals:
        for cy in cy_vals:
            for radius in r_vals:
                score = candidate_score(lab, int(cx), int(cy), int(radius))
                if best is None or score > best[0]:
                    best = (score, int(cx), int(cy), int(radius))

    if best is None or best[0] < 3.0:
        cx, cy, radius = default_cx, default_cy, default_r
        score = 0.0
        source = "auto_default"
    else:
        score, cx_s, cy_s, r_s = best
        cx = int(round(cx_s / scale))
        cy = int(round(cy_s / scale))
        radius = int(round(r_s / scale))
        source = "auto_contrast"

    bbox = clamp_bbox((cx - radius, cy - radius, cx + radius, cy + radius), w, h)
    return {"bbox": bbox, "confidence": min(0.99, max(0.35, float(score) / 30.0)), "source": source}


def load_yolo_model() -> Any | None:
    if not YOLO_MODEL_PATH.exists() or not yolo_available():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(YOLO_MODEL_PATH))
    except Exception:
        return None


def yolo_detect_spot(img_bgr: np.ndarray, model: Any) -> dict[str, Any] | None:
    try:
        results = model.predict(source=img_bgr, imgsz=640, conf=0.20, verbose=False)
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        h, w = img_bgr.shape[:2]
        center = np.array([w / 2.0, h / 2.0])
        best = None
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            conf = float(box.conf.cpu().numpy().reshape(-1)[0])
            bx_center = np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])
            dist = np.linalg.norm((bx_center - center) / np.array([w, h]))
            score = conf - 0.35 * dist
            if best is None or score > best[0]:
                best = (score, conf, tuple(float(v) for v in xyxy))
        if best is None:
            return None
        _, conf, bbox = best
        return {"bbox": clamp_bbox(bbox, w, h), "confidence": conf, "source": "yolo"}
    except Exception:
        return None


def detect_spot(img_bgr: np.ndarray, yolo_model: Any | None = None) -> dict[str, Any]:
    if yolo_model is not None:
        yolo_result = yolo_detect_spot(img_bgr, yolo_model)
        if yolo_result is not None:
            return yolo_result
    return auto_detect_spot(img_bgr)


def make_masks(shape: tuple[int, int], bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    radius = min(x2 - x1, y2 - y1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    spot = dist <= radius * 0.68
    annulus = (dist >= radius * 1.15) & (dist <= radius * 1.45)
    border = np.zeros((h, w), dtype=bool)
    pad = max(20, int(round(min(h, w) * 0.055)))
    border[:pad, :] = True
    border[-pad:, :] = True
    border[:, :pad] = True
    border[:, -pad:] = True
    background = annulus | border
    return spot, annulus, background


def image_quality_checks(img_bgr: np.ndarray, bbox: tuple[float, float, float, float], source: str, confidence: float) -> list[str]:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    reasons: list[str] = []
    if source == "yolo" and confidence < 0.20:
        reasons.append("YOLO 圆斑置信度过低")
    if not (0.35 * w <= cx <= 0.68 * w and 0.35 * h <= cy <= 0.72 * h):
        reasons.append("圆斑位置明显偏离中心")
    if bw < 0.26 * min(w, h) or bh < 0.26 * min(w, h) or bw > 0.86 * min(w, h) or bh > 0.86 * min(w, h):
        reasons.append("圆斑尺寸异常")
    spot_mask, _, _ = make_masks((h, w), bbox)
    if spot_mask.sum() < 500:
        reasons.append("有效圆斑像素过少")
        return reasons
    spot = img_bgr[spot_mask]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(np.var(lap[spot_mask]))
    mean_v = float(np.mean(spot))
    saturated = float(np.mean((spot <= 3) | (spot >= 252)))
    if mean_v < 18:
        reasons.append("圆斑区域过暗")
    if mean_v > 242:
        reasons.append("圆斑区域过曝")
    if saturated > 0.12:
        reasons.append("圆斑区域过曝或欠曝像素过多")
    # These close-up color patches are naturally soft; reject only extreme defocus.
    if blur_score < 0.20:
        reasons.append("图片严重模糊")
    return reasons


def vector_stats(values: np.ndarray) -> list[float]:
    return [
        *np.mean(values, axis=0).tolist(),
        *np.median(values, axis=0).tolist(),
        *np.std(values, axis=0).tolist(),
        *np.percentile(values, 10, axis=0).tolist(),
        *np.percentile(values, 90, axis=0).tolist(),
    ]


def extract_features(img_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    spot_mask, _, bg_mask = make_masks((h, w), bbox)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    bg_rgb = img_rgb[bg_mask]
    bg_med = np.median(bg_rgb, axis=0)
    scale = bg_med.mean() / (bg_med + 1e-6)
    corrected = np.clip(img_rgb * scale, 0, 255).astype(np.uint8)

    spaces = [
        corrected.astype(np.float32),
        cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32),
        cv2.cvtColor(corrected, cv2.COLOR_RGB2HSV).astype(np.float32),
    ]

    features: list[float] = []
    summary: list[float] = []
    for space in spaces:
        spot = space[spot_mask]
        bg = space[bg_mask]
        spot_mean = np.mean(spot, axis=0)
        bg_mean = np.mean(bg, axis=0)
        spot_med = np.median(spot, axis=0)
        bg_med_space = np.median(bg, axis=0)
        features.extend(vector_stats(spot))
        features.extend((spot_mean - bg_mean).tolist())
        features.extend((spot_med - bg_med_space).tolist())
        features.extend((spot_mean / (bg_mean + 1e-6)).tolist())
        summary.extend((spot_mean - bg_mean).tolist())

    spot_rgb = corrected[spot_mask].astype(np.float32)
    bg_rgb_corr = corrected[bg_mask].astype(np.float32)
    spot_mean = np.mean(spot_rgb, axis=0)
    bg_mean = np.mean(bg_rgb_corr, axis=0)
    chroma = spot_mean / (spot_mean.sum() + 1e-6)
    absorbance = np.log((bg_mean + 1.0) / (spot_mean + 1.0))
    features.extend(chroma.tolist())
    features.extend(absorbance.tolist())
    summary.extend(chroma.tolist())
    summary.extend(absorbance.tolist())
    features.append(float(np.linalg.norm(absorbance)))
    summary.append(float(np.linalg.norm(absorbance)))
    return np.array(features, dtype=np.float32), np.array(summary, dtype=np.float32)


@dataclass
class Sample:
    name: str
    concentration: float
    img_bgr: np.ndarray
    image_bytes: bytes | None = None
    origin: str = "new"
    record_id: str = ""
    user_name: str = ""
    bbox: tuple[float, float, float, float] | None = None
    detector_source: str = ""
    detector_confidence: float = 0.0
    features: np.ndarray | None = None
    summary: np.ndarray | None = None
    rejected: bool = False
    reason: str = ""


def sample_record_id(sample: Sample) -> str:
    if sample.record_id:
        return sample.record_id
    digest = hashlib.sha1()
    digest.update(sample.name.encode("utf-8", errors="ignore"))
    digest.update(str(sample.concentration).encode("ascii", errors="ignore"))
    if sample.image_bytes:
        digest.update(sample.image_bytes)
    else:
        ok, encoded = cv2.imencode(".jpg", sample.img_bgr)
        digest.update(encoded.tobytes() if ok else sample.img_bgr.tobytes())
    sample.record_id = digest.hexdigest()[:20]
    return sample.record_id


def mark_reject(sample: Sample, reason: str) -> None:
    sample.rejected = True
    sample.reason = reason


def initial_process_samples(samples: list[Sample]) -> None:
    yolo_model = load_yolo_model()
    for sample in samples:
        if sample.rejected:
            continue
        try:
            detection = detect_spot(sample.img_bgr, yolo_model)
            sample.bbox = detection["bbox"]
            sample.detector_source = detection["source"]
            sample.detector_confidence = float(detection["confidence"])
            reasons = image_quality_checks(sample.img_bgr, sample.bbox, sample.detector_source, sample.detector_confidence)
            if reasons:
                mark_reject(sample, "；".join(reasons))
                continue
            sample.features, sample.summary = extract_features(sample.img_bgr, sample.bbox)
        except Exception as exc:
            mark_reject(sample, f"图像处理失败：{exc}")


def reject_duplicate_samples(samples: list[Sample]) -> None:
    seen: set[str] = set()
    for sample in samples:
        record_id = sample_record_id(sample)
        if record_id in seen:
            mark_reject(sample, "与已有训练图片重复，未重复加入训练集")
        else:
            seen.add(record_id)


def apply_existing_model_label_check(new_samples: list[Sample]) -> None:
    artifact = load_artifact()
    if artifact is None:
        return
    model = artifact.get("regressor")
    levels = [float(v) for v in artifact.get("levels", [])]
    if model is None or not levels:
        return
    low = min(levels)
    high = max(levels)
    for sample in new_samples:
        if sample.rejected or sample.features is None:
            continue
        if not (low <= sample.concentration <= high):
            continue
        try:
            estimate = float(safe_predict(model, sample.features.reshape(1, -1))[0])
        except Exception:
            continue
        residual = abs(estimate - sample.concentration)
        threshold = max(0.08, abs(sample.concentration) * 0.15)
        if residual > threshold:
            mark_reject(sample, f"与现有模型偏差过大（现有模型预测 {estimate:.3f} mg/L，阈值 {threshold:.3f}）")


def load_archived_samples() -> list[Sample]:
    if not ARCHIVE_PATH.exists():
        return []
    try:
        archive = joblib.load(ARCHIVE_PATH)
    except Exception:
        return []
    samples: list[Sample] = []
    for record in archive.get("records", []):
        path = Path(record.get("image_path", ""))
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        samples.append(
            Sample(
                name=record.get("name", path.name),
                concentration=float(record["concentration"]),
                img_bgr=img,
                origin="archive",
                record_id=record.get("record_id", path.stem),
                user_name=record.get("user_name", ""),
            )
        )
    return samples


def clear_archive() -> None:
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if YOLO_MODEL_PATH.exists():
        YOLO_MODEL_PATH.unlink()
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    if TRAINING_IMAGE_DIR.exists():
        shutil.rmtree(TRAINING_IMAGE_DIR)


def save_training_archive(samples: list[Sample]) -> dict[str, Any]:
    accepted = [sample for sample in samples if not sample.rejected and sample.features is not None]
    if TRAINING_IMAGE_DIR.exists():
        shutil.rmtree(TRAINING_IMAGE_DIR)
    TRAINING_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(accepted):
        record_id = sample_record_id(sample)
        path = TRAINING_IMAGE_DIR / f"{idx:05d}_{record_id}.jpg"
        cv2.imwrite(str(path), sample.img_bgr)
        records.append(
            {
                "record_id": record_id,
                "name": sample.name,
                "concentration": float(sample.concentration),
                "user_name": sample.user_name,
                "image_path": str(path),
                "bbox": [float(v) for v in sample.bbox] if sample.bbox else None,
                "detector_source": sample.detector_source,
                "detector_confidence": float(sample.detector_confidence),
                "saved_at": now_stamp(),
            }
        )
    archive = {"updated_at": now_stamp(), "records": records}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(archive, ARCHIVE_PATH)
    return {"updated_at": archive["updated_at"], "count": len(records)}


def apply_group_outlier_filter(samples: list[Sample]) -> None:
    by_conc: dict[float, list[Sample]] = {}
    for sample in samples:
        if not sample.rejected and sample.summary is not None:
            by_conc.setdefault(sample.concentration, []).append(sample)
    for concentration, group in by_conc.items():
        if len(group) < 5:
            continue
        matrix = np.vstack([sample.summary for sample in group])
        med = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - med), axis=0)
        mad = np.where(mad < 1e-5, 1.0, mad)
        robust = np.median(np.abs((matrix - med) / mad), axis=1)
        q75 = float(np.percentile(robust, 75))
        threshold = max(8.0, q75 * 2.8)
        for sample, score in zip(group, robust):
            if score > threshold:
                mark_reject(sample, f"同浓度组颜色特征偏离过大（robust={score:.2f}）")


def make_regressors(n_samples: int, n_features: int) -> dict[str, Any]:
    pls_components = max(1, min(6, n_samples - 2, n_features))
    return {
        "ExtraTrees": ExtraTreesRegressor(random_state=42, n_estimators=240, min_samples_leaf=2, n_jobs=-1),
        "SVR": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.03)),
        "PLS": make_pipeline(StandardScaler(), PLSRegression(n_components=pls_components)),
    }


def safe_predict(model: Any, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    return np.asarray(pred, dtype=float).reshape(-1)


def evaluate_models(x: np.ndarray, y: np.ndarray, class_ids: np.ndarray) -> tuple[str, Any, dict[str, Any], np.ndarray]:
    counts = np.bincount(class_ids)
    n_splits = int(min(5, counts[counts > 0].min())) if counts.size else 0
    regressors = make_regressors(len(y), x.shape[1])
    levels = np.array(sorted(set(float(v) for v in y)))
    metrics: dict[str, Any] = {}
    best_name = ""
    best_model: Any | None = None
    best_pred: np.ndarray | None = None
    best_mae = float("inf")

    for name, model in regressors.items():
        try:
            if n_splits >= 2 and len(levels) >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                pred = cross_val_predict(model, x, y, cv=cv.split(x, class_ids))
            else:
                fitted = clone(model).fit(x, y)
                pred = safe_predict(fitted, x)
            pred = np.clip(pred, float(np.min(y)), float(np.max(y)))
            mae = float(mean_absolute_error(y, pred))
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            r2 = float(r2_score(y, pred)) if len(set(y.tolist())) > 1 else 1.0
            nearest = levels[np.argmin(np.abs(pred[:, None] - levels[None, :]), axis=1)]
            nearest_acc = float(np.mean(np.isclose(nearest, y)))
            metrics[name] = {"mae": mae, "rmse": rmse, "r2": r2, "nearest_accuracy": nearest_acc}
            if mae < best_mae:
                best_mae = mae
                best_name = name
                best_model = clone(model).fit(x, y)
                best_pred = pred
        except Exception as exc:
            metrics[name] = {"error": str(exc)}
    if best_model is None or best_pred is None:
        raise RuntimeError("所有回归模型训练失败")
    return best_name, best_model, metrics, best_pred


def apply_cv_residual_filter(samples: list[Sample]) -> dict[str, Any] | None:
    accepted = [sample for sample in samples if not sample.rejected and sample.features is not None]
    if len(accepted) < 18 or len({sample.concentration for sample in accepted}) < 3:
        return None
    x = np.vstack([sample.features for sample in accepted])
    y = np.array([sample.concentration for sample in accepted], dtype=float)
    levels = sorted(set(y.tolist()))
    class_ids = np.array([levels.index(v) for v in y], dtype=int)
    try:
        _, _, metrics, pred = evaluate_models(x, y, class_ids)
    except Exception:
        return None
    by_conc_counts: dict[float, int] = {}
    for sample in accepted:
        by_conc_counts[sample.concentration] = by_conc_counts.get(sample.concentration, 0) + 1
    for sample, estimate in zip(accepted, pred):
        residual = abs(float(estimate) - sample.concentration)
        threshold = max(0.08, abs(sample.concentration) * 0.15)
        if residual > threshold and by_conc_counts.get(sample.concentration, 0) > 3:
            mark_reject(sample, f"交叉验证残差过大（预测 {estimate:.3f} mg/L，阈值 {threshold:.3f}）")
            by_conc_counts[sample.concentration] -= 1
    return metrics


def group_warnings(samples: list[Sample], strict_min_count: bool = True) -> list[dict[str, Any]]:
    concentrations = sorted(set(sample.concentration for sample in samples))
    warnings: list[dict[str, Any]] = []
    for concentration in concentrations:
        group = [sample for sample in samples if sample.concentration == concentration]
        accepted = [sample for sample in group if not sample.rejected]
        rejected = len(group) - len(accepted)
        ratio = rejected / max(1, len(group))
        if strict_min_count and len(accepted) < 3:
            warnings.append(
                {
                    "concentration": concentration,
                    "message": f"{concentration:g} mg/L 可用图片少于 3 张",
                    "accepted": len(accepted),
                    "rejected": rejected,
                }
            )
        elif ratio > 0.40:
            warnings.append(
                {
                    "concentration": concentration,
                    "message": f"{concentration:g} mg/L 被拒绝比例超过 40%",
                    "accepted": len(accepted),
                    "rejected": rejected,
                }
            )
    return warnings


def save_uploaded_training_images(samples: list[Sample], root: Path) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    accepted = [sample for sample in samples if not sample.rejected and sample.bbox is not None]
    if not accepted:
        return
    for idx, sample in enumerate(accepted):
        split = "val" if idx % 5 == 0 else "train"
        img_path = images_dir / split / f"spot_{idx:04d}.jpg"
        label_path = labels_dir / split / f"spot_{idx:04d}.txt"
        cv2.imwrite(str(img_path), sample.img_bgr)
        h, w = sample.img_bgr.shape[:2]
        x1, y1, x2, y2 = sample.bbox or (0, 0, w, h)
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        label_path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")
    yaml_path = root / "spot.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: spot\n",
        encoding="utf-8",
    )


def train_yolo_if_possible(samples: list[Sample]) -> dict[str, Any]:
    result = {"attempted": False, "available": yolo_available(), "trained": False, "message": ""}
    if not result["available"]:
        result["message"] = "ultralytics 未安装，使用自动圆斑定位"
        return result
    accepted = [sample for sample in samples if not sample.rejected and sample.bbox is not None]
    if len(accepted) < 12:
        result["message"] = "可用样本不足，跳过 YOLO 训练"
        return result
    result["attempted"] = True
    dataset_root = WORK_DIR / "yolo_spot_dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    save_uploaded_training_images(samples, dataset_root)
    try:
        from ultralytics import YOLO

        base_model = str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH.exists() else "yolov8n.yaml"
        model = YOLO(base_model)
        run = model.train(
            data=str(dataset_root / "spot.yaml"),
            epochs=1,
            imgsz=416,
            batch=16,
            patience=1,
            verbose=False,
            project=str(WORK_DIR / "yolo_runs"),
            name="spot",
            exist_ok=True,
            pretrained=False,
            plots=False,
            val=False,
            workers=0,
        )
        weights = Path(run.save_dir) / "weights" / "best.pt"
        if not weights.exists():
            weights = Path(run.save_dir) / "weights" / "last.pt"
        if weights.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights, YOLO_MODEL_PATH)
            mode = "继续训练" if Path(base_model).exists() else "快速初训"
            result.update({"trained": True, "message": f"YOLO 圆斑检测器已{mode}"})
        else:
            result["message"] = "YOLO 训练完成但未找到权重文件，使用自动圆斑定位"
    except Exception as exc:
        result["message"] = f"YOLO 训练失败，使用自动圆斑定位：{exc}"
    return result


def rejected_payload(samples: list[Sample]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.rejected:
            continue
        overlay = ""
        if len(payload) < MAX_REJECT_THUMBS:
            overlay = encode_overlay(sample.img_bgr, sample.bbox, "rejected")
        payload.append(
            {
                "name": sample.name,
                "user_name": sample.user_name,
                "concentration": sample.concentration,
                "reason": sample.reason,
                "overlay": overlay,
            }
        )
    return payload


def process_payload(samples: list[Sample], only_new: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = [sample for sample in samples if (sample.origin == "new" or not only_new)]
    for sample in source[:MAX_PROCESS_THUMBS]:
        rows.append(
            {
                "name": sample.name,
                "user_name": sample.user_name,
                "concentration": sample.concentration,
                "origin": sample.origin,
                "status": "rejected" if sample.rejected else "accepted",
                "reason": sample.reason,
                "detector_source": sample.detector_source,
                "detector_confidence": float(sample.detector_confidence),
                "bbox": [float(v) for v in sample.bbox] if sample.bbox else None,
                "overlay": encode_overlay(sample.img_bgr, sample.bbox, "rejected" if sample.rejected else "ok"),
            }
        )
    return rows


def train_pipeline(new_samples: list[Sample], reset_archive_flag: bool = False) -> dict[str, Any]:
    if not new_samples:
        raise ValueError("没有收到训练图片")
    if reset_archive_flag:
        clear_archive()
    archived_samples = load_archived_samples()
    samples = archived_samples + new_samples
    reject_duplicate_samples(samples)
    initial_process_samples(samples)
    apply_existing_model_label_check(new_samples)
    apply_group_outlier_filter(samples)
    preliminary_metrics = apply_cv_residual_filter(samples)
    new_accepted = [sample for sample in new_samples if not sample.rejected and sample.features is not None]
    if not new_accepted:
        accepted = [sample for sample in samples if not sample.rejected and sample.features is not None]
        return {
            "ok": False,
            "error": "新增训练样本全部被拒绝，模型未更新",
            "new_count": len(new_samples),
            "archive_count_before": len(archived_samples),
            "accepted_count": len(accepted),
            "rejected_count": len(samples) - len(accepted),
            "process_samples": process_payload(samples),
            "rejected_samples": rejected_payload(samples),
            "preliminary_metrics": preliminary_metrics or {},
        }
    warnings = group_warnings(samples, strict_min_count=not bool(archived_samples))
    accepted = [sample for sample in samples if not sample.rejected and sample.features is not None]
    if warnings:
        return {
            "ok": False,
            "error": "训练集未通过浓度组质控",
            "group_warnings": warnings,
            "new_count": len(new_samples),
            "archive_count_before": len(archived_samples),
            "accepted_count": len(accepted),
            "rejected_count": len(samples) - len(accepted),
            "process_samples": process_payload(samples),
            "rejected_samples": rejected_payload(samples),
            "preliminary_metrics": preliminary_metrics or {},
        }
    if len(accepted) < 6 or len({sample.concentration for sample in accepted}) < 2:
        raise ValueError("可用训练样本不足，至少需要 2 个浓度且总样本不少于 6 张")

    yolo_info = train_yolo_if_possible(samples)
    x = np.vstack([sample.features for sample in accepted])
    y = np.array([sample.concentration for sample in accepted], dtype=float)
    levels = sorted(set(float(v) for v in y))
    class_ids = np.array([levels.index(float(v)) for v in y], dtype=int)
    best_name, best_model, metrics, pred = evaluate_models(x, y, class_ids)
    pred = np.clip(pred, min(levels), max(levels))
    per_concentration: list[dict[str, Any]] = []
    for level in levels:
        mask = np.isclose(y, level)
        errors = np.abs(pred[mask] - y[mask])
        per_concentration.append(
            {
                "concentration": level,
                "count": int(mask.sum()),
                "mae": float(np.mean(errors)),
                "bias": float(np.mean(pred[mask] - y[mask])),
                "std": float(np.std(pred[mask] - y[mask])),
            }
        )

    archive_info = save_training_archive(samples)
    artifact = {
        "created_at": now_stamp(),
        "regressor_name": best_name,
        "regressor": best_model,
        "levels": levels,
        "range": [float(min(levels)), float(max(levels))],
        "metrics": metrics,
        "per_concentration": per_concentration,
        "accepted_count": len(accepted),
        "rejected_count": len(samples) - len(accepted),
        "new_count": len(new_samples),
        "archive_count_before": len(archived_samples),
        "archive_count_after": archive_info["count"],
        "feature_count": int(x.shape[1]),
        "yolo": yolo_info,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    return {
        "ok": True,
        "model": {
            "created_at": artifact["created_at"],
            "regressor_name": best_name,
            "levels": levels,
            "range": artifact["range"],
            "accepted_count": artifact["accepted_count"],
            "rejected_count": artifact["rejected_count"],
            "new_count": artifact["new_count"],
            "archive_count_before": artifact["archive_count_before"],
            "archive_count_after": artifact["archive_count_after"],
            "feature_count": artifact["feature_count"],
        },
        "metrics": metrics,
        "best_model": best_name,
        "per_concentration": per_concentration,
        "yolo": yolo_info,
        "process_samples": process_payload(samples),
        "rejected_samples": rejected_payload(samples),
    }


def load_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def model_status() -> dict[str, Any]:
    artifact = load_artifact()
    archive_count = len(load_archived_samples())
    if artifact is None:
        return {
            "ok": True,
            "has_model": False,
            "archive_count": archive_count,
            "yolo_available": yolo_available(),
            "yolo_model_exists": YOLO_MODEL_PATH.exists(),
            "message": "尚未训练模型",
        }
    return {
        "ok": True,
        "has_model": True,
        "created_at": artifact.get("created_at"),
        "regressor_name": artifact.get("regressor_name"),
        "levels": artifact.get("levels", []),
        "range": artifact.get("range", []),
        "accepted_count": artifact.get("accepted_count", 0),
        "rejected_count": artifact.get("rejected_count", 0),
        "archive_count": archive_count,
        "archive_count_after": artifact.get("archive_count_after", archive_count),
        "new_count": artifact.get("new_count", 0),
        "metrics": artifact.get("metrics", {}),
        "yolo": artifact.get("yolo", {}),
        "yolo_available": yolo_available(),
        "yolo_model_exists": YOLO_MODEL_PATH.exists(),
    }


def predict_image(img_bgr: np.ndarray, name: str = "image") -> dict[str, Any]:
    artifact = load_artifact()
    if artifact is None:
        raise ValueError("尚未训练模型，请先在训练模式中训练")
    yolo_model = load_yolo_model()
    detection = detect_spot(img_bgr, yolo_model)
    bbox = detection["bbox"]
    reasons = image_quality_checks(img_bgr, bbox, detection["source"], float(detection["confidence"]))
    if reasons:
        return {
            "ok": True,
            "name": name,
            "qc_status": "fail",
            "qc_reasons": reasons,
            "spot_bbox": [float(v) for v in bbox],
            "detector": detection,
            "overlay": encode_overlay(img_bgr, bbox, "qc fail"),
        }
    features, _ = extract_features(img_bgr, bbox)
    pred = float(safe_predict(artifact["regressor"], features.reshape(1, -1))[0])
    levels = np.array(artifact.get("levels", []), dtype=float)
    low, high = artifact.get("range", [float(np.min(levels)), float(np.max(levels))])
    pred_clipped = float(np.clip(pred, low, high))
    nearest = float(levels[np.argmin(np.abs(levels - pred_clipped))])
    extrapolation = pred < low or pred > high
    distance = float(abs(pred_clipped - nearest))
    return {
        "ok": True,
        "name": name,
        "qc_status": "ok" if not extrapolation else "warning",
        "qc_reasons": ["预测值超出训练浓度范围，已裁剪到训练范围"] if extrapolation else [],
        "pred_mg_L": pred_clipped,
        "raw_pred_mg_L": pred,
        "nearest_standard_mg_L": nearest,
        "distance_to_standard": distance,
        "spot_bbox": [float(v) for v in bbox],
        "detector": detection,
        "model": {
            "created_at": artifact.get("created_at"),
            "regressor_name": artifact.get("regressor_name"),
            "range": artifact.get("range"),
        },
        "overlay": encode_overlay(img_bgr, bbox, "ok" if not extrapolation else "warning"),
    }


def parse_training_request(form: cgi.FieldStorage) -> list[Sample]:
    samples: list[Sample] = []
    samples_raw = form.getfirst("samples")
    if samples_raw:
        sample_defs = json.loads(samples_raw)
        for item in sample_defs:
            sample_id = str(item.get("id", ""))
            concentration = float(item["concentration"])
            user_name = str(item.get("user_name", "")).strip()
            if not user_name:
                raise ValueError("每张训练图片都必须填写用户姓名")
            field = form[f"image_{sample_id}"] if f"image_{sample_id}" in form else None
            if field is None or isinstance(field, list) or not getattr(field, "filename", ""):
                continue
            blob = field.file.read()
            if not blob:
                continue
            img = read_image_bytes(blob)
            samples.append(
                Sample(
                    name=Path(field.filename).name,
                    concentration=concentration,
                    img_bgr=img,
                    image_bytes=blob,
                    user_name=user_name,
                )
            )
        return samples

    groups_raw = form.getfirst("groups")
    if not groups_raw:
        raise ValueError("缺少训练样本信息")
    groups = json.loads(groups_raw)
    for group in groups:
        group_id = str(group.get("id", ""))
        concentration = float(group["concentration"])
        user_name = str(group.get("user_name", "")).strip()
        fields = form.getlist(f"images_{group_id}")
        for field in fields:
            if not getattr(field, "filename", ""):
                continue
            blob = field.file.read()
            if not blob:
                continue
            img = read_image_bytes(blob)
            samples.append(Sample(name=Path(field.filename).name, concentration=concentration, img_bgr=img, image_bytes=blob, user_name=user_name))
    return samples


class ChromiumHandler(SimpleHTTPRequestHandler):
    server_version = "ChromiumColorimetry/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (now_stamp(), fmt % args))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            if not INDEX_PATH.exists():
                error_response(self, "index.html 不存在", HTTPStatus.NOT_FOUND)
                return
            data = INDEX_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/model/status":
            json_response(self, model_status())
            return
        if self.path.startswith("/static/"):
            path = APP_DIR / self.path.lstrip("/")
            if path.exists() and path.is_file():
                data = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        error_response(self, "Not found", HTTPStatus.NOT_FOUND)

    def parse_multipart(self) -> cgi.FieldStorage:
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
        )

    def do_POST(self) -> None:
        try:
            if self.path == "/api/train":
                form = self.parse_multipart()
                reset_archive_flag = form.getfirst("reset") == "1"
                if reset_archive_flag and form.getfirst("admin_password") != ADMIN_PASSWORD:
                    raise ValueError("管理员密码错误，不能清空历史训练集")
                samples = parse_training_request(form)
                payload = train_pipeline(samples, reset_archive_flag=reset_archive_flag)
                json_response(self, payload, status=HTTPStatus.OK if payload.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/api/predict":
                form = self.parse_multipart()
                field = form["image"] if "image" in form else None
                if field is None or not getattr(field, "filename", ""):
                    raise ValueError("请上传预测图片")
                blob = field.file.read()
                img = read_image_bytes(blob)
                payload = predict_image(img, Path(field.filename).name)
                json_response(self, payload)
                return
            error_response(self, "Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, str(exc), HTTPStatus.BAD_REQUEST)


def train_from_folder(folder: Path, reset_archive_flag: bool = False) -> dict[str, Any]:
    samples: list[Sample] = []
    for concentration_dir in sorted(folder.iterdir()):
        if not concentration_dir.is_dir():
            continue
        name = concentration_dir.name.replace("：", ":")
        if "mg" not in name:
            continue
        try:
            concentration = float(name.split("mg")[0])
        except ValueError:
            continue
        for path in sorted(concentration_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            samples.append(Sample(name=path.name, concentration=concentration, img_bgr=img, image_bytes=path.read_bytes(), user_name="initial"))
    return train_pipeline(samples, reset_archive_flag=reset_archive_flag)


def run_self_test(folder: Path) -> None:
    print(f"Training from {folder} ...")
    result = train_from_folder(folder, reset_archive_flag=True)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rejected_samples", "process_samples"}}, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)
    first = next(path for path in sorted(folder.glob("*/*")) if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    img = cv2.imread(str(first))
    pred = predict_image(img, first.name)
    print(json.dumps({k: v for k, v in pred.items() if k != "overlay"}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chromium concentration local web tool")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--self-test", action="store_true", help="Train from ./样品图片 and run one prediction")
    parser.add_argument("--train-folder", default=str(APP_DIR / "样品图片"))
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        run_self_test(Path(args.train_folder))
        return

    server = ThreadingHTTPServer((args.host, args.port), ChromiumHandler)
    print(f"Chromium colorimetry tool running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
