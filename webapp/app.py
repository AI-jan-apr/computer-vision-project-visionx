"""
VisionX FastAPI server — رفع صورة وتحليلها.

تشغيل من جذر المشروع:
  uvicorn webapp.app:app --reload --host 127.0.0.1 --port 8000

تحذير (ويندوز): افتح في المتصفح http://127.0.0.1:8000
وليس http://localhost:8000 — أحياناً localhost يتصل بـ IPv6 (::1) بينما
الخادم يستمع على IPv4 فقط فيتعطل الاتصال أو يتأخر. يمكنك أيضاً تشغيل run_web.bat
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from main import attach_db_comparison_to_summary, run_pipeline
from utils.config import DEFAULT_DB_PATH, PROJECT_ROOT
from utils.plate_reader import is_plate_reader_loaded

logger = logging.getLogger(__name__)

WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
INTERFACE_HTML = PROJECT_ROOT / "interface" / "raqeeb.html"
SESSION_ROOT = PROJECT_ROOT / "outputs" / "web_sessions"

ALLOWED_UPLOAD_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_RESULT_FILES = frozenset(
    {
        "00_overview.jpg",
        "01_vehicle_annotated.jpg",
        "02_vehicle_crop.jpg",
        "03_plate_on_roi.jpg",
        "04_plate_crop.jpg",
        "05_plate_enhanced.png",
    }
)

SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")

app = FastAPI(title="VisionX", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _user_messages(summary: dict, comp: dict | None) -> list[dict]:
    """رسائل عربية مختصرة للواجهة."""
    out: list[dict] = []

    if not summary.get("vehicle"):
        out.append(
            {
                "level": "warning",
                "text": "لم يُكتشف جسم مركبة واضح في الصورة. جرّب صورة أوضح للسيارة كاملة.",
            }
        )
    else:
        out.append({"level": "success", "text": "تم اكتشاف المركبة."})

    if not summary.get("plate"):
        out.append(
            {
                "level": "warning",
                "text": "لم تُكتشف لوحة في منطقة المركبة.",
            }
        )
    else:
        out.append({"level": "success", "text": "تم اكتشاف اللوحة."})

    pr = summary.get("plate_read") or {}
    if pr.get("final"):
        out.append(
            {
                "level": "info",
                "text": f"قراءة اللوحة (نموذج): {pr.get('final')}",
            }
        )
    elif summary.get("plate") and not pr.get("final"):
        out.append(
            {
                "level": "info",
                "text": "لم تُقرأ اللوحة نصياً (عطّلت قراءة Qwen أو فشلت). يمكنك تفعيلها من الخيارات.",
            }
        )

    if comp is None:
        out.append(
            {
                "level": "info",
                "text": "لا يوجد صف مطابق في قاعدة البيانات لهذه الصورة أو لرقم اللوحة المقروء. استورد Excel أو اربط اسم الملف باللوحة عبر db_cli.",
            }
        )
    elif comp.get("registry_consistent"):
        out.append(
            {
                "level": "success",
                "text": "البيانات المستخرجة تطابق السجل المسجّل (لوحة / ماركة / موديل / لون حسب التوفر).",
            }
        )
    else:
        fields = comp.get("mismatch_fields") or []
        out.append(
            {
                "level": "danger",
                "text": "تنبيه: هناك اختلاف عن السجل الرسمي في: "
                + ", ".join(fields)
                + ". راجع الصورة أو التسجيل (قد يكون خطأ قراءة وليس بالضرورة سرقة).",
            }
        )

    return out


@app.get("/api/health")
def health() -> dict:
    """للتحقق السريع أن الخادم يعمل وحالة تحميل Qwen (بعد أول قراءة لوحة)."""
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {
        "ok": True,
        "cuda_available": cuda,
        "qwen_plate_reader_loaded": is_plate_reader_loaded(),
    }


@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    if not INTERFACE_HTML.is_file():
        return HTMLResponse(
            "<p>Missing interface/raqeeb.html</p>",
            status_code=500,
            headers=_no_cache_headers(),
        )
    body = INTERFACE_HTML.read_text(encoding="utf-8")
    return HTMLResponse(content=body, headers=_no_cache_headers())


def _no_cache_headers() -> dict[str, str]:
    """يمنع المتصفح من تخزين الصفحة مؤقتاً فيطالع دائماً أحدث raqeeb.html بعد التعديل."""
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }


@app.get("/api/results/{session_id}/{filename}")
def get_result_file(session_id: str, filename: str) -> FileResponse:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session")
    if filename not in ALLOWED_RESULT_FILES:
        raise HTTPException(status_code=404, detail="File not allowed")
    path = SESSION_ROOT / session_id / filename
    path = path.resolve()
    if not str(path).startswith(str(SESSION_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    media = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media)


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    skip_qwen: bool = Query(
        False,
        description="إذا true يتخطى Qwen (لا نص لوحة → لا مطابقة باللوحة في DB). الافتراضي: تشغيل Qwen 2.5-VL",
    ),
    skip_brand: bool = Query(False),
    skip_color: bool = Query(False),
    compare_db: bool = Query(True, description="مقارنة مع قاعدة SQLite"),
    db_path: str | None = Query(None, description="مسار اختياري لملف .db"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="لم يُرسل اسم ملف")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail="امتداد غير مدعوم. استخدم jpg أو png أو webp.",
        )

    session_id = uuid.uuid4().hex
    out_dir = SESSION_ROOT / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = out_dir / f"input{suffix}"
    try:
        with saved.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    finally:
        await file.close()

    db = Path(db_path) if db_path else DEFAULT_DB_PATH

    t0 = time.perf_counter()
    logger.info(
        "VisionX: بدء التحليل session=%s skip_qwen=%s",
        session_id,
        skip_qwen,
    )
    try:
        summary = run_pipeline(
            saved,
            out_dir,
            skip_brand=skip_brand,
            skip_color=skip_color,
            skip_qwen=skip_qwen,
            quiet=True,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    comp = None
    if compare_db:
        comp = attach_db_comparison_to_summary(summary, file.filename, db_path=db)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    base = f"/api/results/{session_id}"
    images = {}
    for name in ALLOWED_RESULT_FILES:
        if (out_dir / name).is_file():
            images[name] = f"{base}/{name}"
    messages = _user_messages(summary, comp)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "VisionX: انتهى التحليل session=%s بعد %s ms",
        session_id,
        elapsed_ms,
    )

    return {
        "session_id": session_id,
        "filename": file.filename,
        "elapsed_ms": elapsed_ms,
        "summary": summary,
        "db_comparison": comp,
        "images": images,
        "messages": messages,
    }
