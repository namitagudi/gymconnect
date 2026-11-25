import os
import re
import tempfile
from flask import Blueprint, request, current_app
from . import pose
from .db import fetch
from .posture_rules import (
    BODY_PART_EXERCISES,
    analyze,
    NEED,
    _EXTRA_NEED,
    normalise,
)
from .security import subscription_required, current_user
from .util import ApiError, RL

bp = Blueprint("posture", __name__, url_prefix="/api/posture")
MAX_IMAGE = 8 * 1024 * 1024
MAX_VIDEO = 25 * 1024 * 1024
MAX_SAMPLED_FRAMES = 12

FAMILY_JOINTS = {
    "chest": ["shoulder", "elbow", "wrist", "hip"],
    "back": ["shoulder", "elbow", "wrist", "hip"],
    "shoulders": ["shoulder", "elbow", "wrist", "hip"],
    "biceps": ["shoulder", "elbow", "wrist"],
    "triceps": ["shoulder", "elbow", "wrist"],
    "legs": ["shoulder", "hip", "knee", "ankle"],
    "abs": ["shoulder", "hip", "knee", "ankle"],
}


def _resolve_exercise(raw_ex: str, body_part: str) -> str:
    if not raw_ex:
        raise ApiError(
            422,
            "invalid",
            "Choose an exercise from the selected body-part list.",
            "exercise",
        )

    ex = normalise(raw_ex)

    if ex not in NEED:
        if body_part not in FAMILY_JOINTS:
            raise ApiError(
                422,
                "invalid",
                "Choose a body part and exercise from Workout Logbook.",
                "exercise",
            )
        NEED[ex] = FAMILY_JOINTS[body_part]
        _EXTRA_NEED[ex] = FAMILY_JOINTS[body_part]

    return ex


def _sniff_video(data: bytes, filename: str) -> bool:
    name = (filename or "").lower()
    has_ext = name.endswith((".mp4", ".webm", ".mov"))
    has_sig = (
        (len(data) >= 8 and data[4:8] == b"ftyp")
        or (len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3")
    )
    return has_ext or has_sig


def _attach_recommendations(res: dict, raw_ex: str, ex: str) -> dict:
    has_problem = (
        res.get("status") != "ok"
        or any(i.get("severity") == "warn" for i in res.get("issues", []))
    )
    if not has_problem:
        res["recommended_videos"] = []
        return res

    def words(value):
        return re.findall(r"[a-z0-9]+", (value or "").lower())

    exercise_words = [w for w in words(raw_ex) if len(w) > 2] or words(ex)
    matches = []
    if exercise_words:
        for video in fetch(
            "SELECT id,title,category,trainer_name,instructions,thumb "
            "FROM videos ORDER BY id DESC"
        ):
            searchable = set(
                words(video.get("title", "") + " " + video.get("instructions", ""))
            )
            if any(word in searchable for word in exercise_words):
                matches.append({
                    "id": video["id"],
                    "title": video["title"],
                    "category": video["category"],
                    "trainer_name": video["trainer_name"],
                    "has_thumb": bool(video.get("thumb")),
                })
            if len(matches) >= 3:
                break

    res["recommended_videos"] = matches
    return res


def _sample_video_frames(video_bytes: bytes, suffix: str, max_frames: int = MAX_SAMPLED_FRAMES):
    try:
        import cv2
    except ImportError as exc:
        raise ApiError(
            500,
            "missing_dependency",
            "OpenCV is required for video posture analysis.",
        ) from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ApiError(
                422,
                "invalid_video",
                "Could not read this video file. Try an MP4 or WebM clip.",
                "video",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames_jpeg = []

        if total_frames > 0:
            step = max(1, total_frames // max_frames)
            target_set = {
                i * step for i in range(max_frames) if i * step < total_frames
            } or {0}
            idx = 0
            while len(frames_jpeg) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in target_set:
                    encoded_ok, buf = cv2.imencode(".jpg", frame)
                    if encoded_ok:
                        frames_jpeg.append(buf.tobytes())
                idx += 1
        else:
            idx = 0
            every = 10
            while len(frames_jpeg) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % every == 0:
                    encoded_ok, buf = cv2.imencode(".jpg", frame)
                    if encoded_ok:
                        frames_jpeg.append(buf.tobytes())
                idx += 1

        cap.release()
        return frames_jpeg
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@bp.get("/catalog")
@subscription_required
def catalog():
    return {"body_parts": BODY_PART_EXERCISES}


@bp.post("/analyze")
@subscription_required
def analyze_photo():
    u = current_user()
    RL.check(f"posture:{u['id']}", 30, 3600)

    raw_ex = (request.form.get("exercise") or "").strip()
    body_part = (request.form.get("body_part") or "").strip().lower()
    ex = _resolve_exercise(raw_ex, body_part)

    f = request.files.get("image")
    if not f:
        raise ApiError(422, "invalid", "Upload a JPG or PNG photo.", "image")

    data = f.read(MAX_IMAGE + 1)
    if len(data) > MAX_IMAGE:
        raise ApiError(413, "too_large", "Photo must be under 8 MB.", "image")

    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        raise ApiError(
            422,
            "invalid",
            "Please upload a JPG or PNG photo (PDFs aren't supported - save the photo as JPG).",
            "image",
        )

    lms, w, h = pose.detect(data, current_app.config["POSE_MODEL_PATH"])
    res = analyze(ex, lms, w, h)
    return _attach_recommendations(res, raw_ex, ex)


@bp.post("/analyze-video")
@subscription_required
def analyze_video():
    u = current_user()
    RL.check(f"posture:{u['id']}", 30, 3600)

    raw_ex = (request.form.get("exercise") or "").strip()
    body_part = (request.form.get("body_part") or "").strip().lower()
    ex = _resolve_exercise(raw_ex, body_part)

    f = request.files.get("video") or request.files.get("file")
    if not f:
        raise ApiError(422, "invalid", "Upload an MP4 or WebM video.", "video")

    data = f.read(MAX_VIDEO + 1)
    if len(data) > MAX_VIDEO:
        raise ApiError(413, "too_large", "Video must be under 25 MB.", "video")

    if not _sniff_video(data, f.filename or ""):
        raise ApiError(
            422,
            "invalid",
            "Please upload an MP4 or WebM video file.",
            "video",
        )

    ext = os.path.splitext((f.filename or "").lower())[1]
    suffix = ext if ext in (".mp4", ".webm", ".mov") else ".mp4"
    frames = _sample_video_frames(data, suffix=suffix, max_frames=MAX_SAMPLED_FRAMES)

    if not frames:
        raise ApiError(
            422,
            "empty_video",
            "No frames could be extracted from the uploaded video.",
            "video",
        )

    model_path = current_app.config["POSE_MODEL_PATH"]
    results = []
    for frame_bytes in frames:
        lms, w, h = pose.detect(frame_bytes, model_path)
        results.append(analyze(ex, lms, w, h))

    ok_results = [r for r in results if r.get("status") == "ok"]
    best = (
        max(ok_results, key=lambda r: r.get("confidence", 0.0))
        if ok_results
        else max(results, key=lambda r: r.get("confidence", 0.0))
    )

    seen_issues = set()
    combined_issues = []
    combined_corrections = []
    for r in (ok_results or results):
        for issue in r.get("issues", []):
            key = (issue.get("severity"), issue.get("title"), issue.get("detail"))
            if key not in seen_issues:
                seen_issues.add(key)
                combined_issues.append(issue)
        for corr in r.get("corrections", []):
            if corr not in combined_corrections:
                combined_corrections.append(corr)

    merged = dict(best)
    merged["single_image"] = False
    merged["sampled_frames"] = len(results)
    merged["valid_frames"] = len(ok_results)
    if combined_issues:
        merged["issues"] = combined_issues
        merged["corrections"] = combined_corrections

    obs = list(best.get("observations") or [])
    obs.insert(
        0,
        f"Analyzed {len(results)} sampled frames ({len(ok_results)} with clear pose visibility).",
    )
    merged["observations"] = obs

    return _attach_recommendations(merged, raw_ex, ex)