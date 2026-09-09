import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from backend.app.config.settings import API_TITLE, API_VERSION, UPLOADS_DIR
from backend.app.database.database import get_connection, init_db
from backend.app.schemas.models import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    DashboardSummaryResponse,
    ExplainRequest,
    ExplainResponse,
    ExtractRequest,
    ExtractResponse,
    PredictRequest,
    PredictResponse,
    ReportSummaryResponse,
    UploadResponse,
)
from backend.app.services.explainability_service import ExplainabilityService
from backend.app.services.extraction_service import ExtractionService
from backend.app.services.pdf_service import PDFService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.rag_service import RAGService
from backend.app.services.settings_service import get_settings, save_settings

app = FastAPI(title=API_TITLE, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "https://medilens-frontend-cevj.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

extraction_service = ExtractionService()
prediction_service = PredictionService()
explainability_service = ExplainabilityService()
pdf_service = PDFService()
rag_service = RAGService()


RADAR_CATEGORY_GROUPS: dict[str, set[str]] = {
    "Hematology": {
        "Hemoglobin",
        "RBC",
        "WBC",
        "Platelets",
        "MCV",
        "MCH",
        "MCHC",
        "RDW",
        "Ferritin",
        "Vitamin B12",
    },
    "Metabolic": {
        "Glucose",
        "HbA1c",
        "Cholesterol",
        "Triglycerides",
        "HDL",
        "LDL",
        "Vitamin D",
    },
    "Renal": {
        "Creatinine",
        "Urea",
        "eGFR",
        "Sodium",
        "Potassium",
        "Chloride",
    },
    "Hepatic": {
        "ALT",
        "AST",
        "ALP",
        "Bilirubin",
    },
    "Endocrine": {
        "TSH",
        "Glucose",
        "HbA1c",
    },
    "Electrolytes": {
        "Sodium",
        "Potassium",
        "Chloride",
    },
}


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------

@app.get("/settings")
def read_settings():
    return get_settings()


@app.put("/settings")
def update_settings(settings: dict):
    save_settings(settings)
    return {"status": "ok", "settings": settings}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(raw_value: Any, default: Any) -> Any:
    if isinstance(raw_value, (list, dict)):
        return raw_value

    if not raw_value:
        return default

    try:
        return json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return default


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, set):
        return sorted(value)

    return str(value)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)

    try:
        return row[key]
    except Exception:
        return default


def _normalize_parameter_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": _row_value(row, "id"),
            "name": _row_value(row, "name"),
            "value": _row_value(row, "value"),
            "unit": _row_value(row, "unit", ""),
            "normalRange": _row_value(row, "normal_range", "")
            or _row_value(row, "normalRange", ""),
            "status": _row_value(row, "status", "normal") or "normal",
        }
        for row in rows
    ]


def _prediction_from_row(
    row: Any,
    *,
    index: int,
    report_id: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": _row_value(
            row,
            "id",
            f"{report_id or 'report'}-risk-{index}",
        ),
        "name": _row_value(row, "name", ""),
        "probability": round(
            float(_row_value(row, "probability", 0) or 0),
            1,
        ),
        "confidence": round(
            float(_row_value(row, "confidence", 0) or 0),
            1,
        ),
        "severity": _row_value(row, "severity", "low") or "low",
        "summary": _row_value(row, "summary", "") or "",
    }

    for field in (
        "predicted_class",
        "model_metrics",
        "shap_values",
        "top_features",
        "feature_importance_visualization_data",
        "human_readable_explanation",
    ):
        value = _row_value(row, field)

        if value is not None:
            payload[field] = value

    shap_summary = _safe_json_loads(
        _row_value(row, "shap_summary"),
        {},
    )

    if isinstance(shap_summary, dict):
        for field in (
            "predicted_class",
            "model_metrics",
            "shap_values",
            "top_features",
            "feature_importance_visualization_data",
            "human_readable_explanation",
        ):
            if field in shap_summary and shap_summary[field] is not None:
                payload[field] = shap_summary[field]

    return payload


def _normalize_prediction_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    return [
        _prediction_from_row(row, index=index)
        for index, row in enumerate(rows)
    ]


def _status_from_predictions(
    predictions: List[Dict[str, Any]],
) -> str:
    if any(
        prediction.get("severity") == "high"
        or float(prediction.get("probability", 0)) >= 70
        for prediction in predictions
    ):
        return "critical"

    if any(
        prediction.get("severity") == "moderate"
        or float(prediction.get("probability", 0)) >= 40
        for prediction in predictions
    ):
        return "attention"

    return "normal"


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

def _build_radar_data(
    report_type: str,
    parameters: List[Dict[str, Any]],
    risk_score: int,
) -> List[Dict[str, Any]]:
    del report_type

    base_score = max(0, 100 - int(risk_score))

    grouped_parameters = {
        subject: []
        for subject in RADAR_CATEGORY_GROUPS
    }

    for parameter in parameters:
        name = str(parameter.get("name", ""))

        for subject, names in RADAR_CATEGORY_GROUPS.items():
            if name in names:
                grouped_parameters[subject].append(parameter)

    def category_value(
        subject: str,
        modifier: int = 0,
    ) -> int:
        matched = grouped_parameters.get(subject, [])

        if not matched:
            return max(
                10,
                min(100, base_score + modifier),
            )

        abnormal_count = sum(
            1
            for parameter in matched
            if parameter.get("status") != "normal"
        )

        normal_count = len(matched) - abnormal_count

        score = (
            base_score
            + modifier
            + normal_count * 3
            - abnormal_count * 12
        )

        return max(
            0,
            min(100, int(round(score))),
        )

    return [
        {
            "subject": "Hematology",
            "value": category_value("Hematology", 6),
            "fullMark": 100,
        },
        {
            "subject": "Metabolic",
            "value": category_value("Metabolic", 2),
            "fullMark": 100,
        },
        {
            "subject": "Renal",
            "value": category_value("Renal", -6),
            "fullMark": 100,
        },
        {
            "subject": "Hepatic",
            "value": category_value("Hepatic", 4),
            "fullMark": 100,
        },
        {
            "subject": "Endocrine",
            "value": category_value("Endocrine", -2),
            "fullMark": 100,
        },
        {
            "subject": "Electrolytes",
            "value": category_value("Electrolytes", 1),
            "fullMark": 100,
        },
    ]


def _build_trend(
    conn,
    patient_name: str,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT report_date, risk_score
        FROM reports
        WHERE patient_name = ?
        ORDER BY created_at ASC
        LIMIT 6
        """,
        (patient_name,),
    ).fetchall()

    trend: List[Dict[str, Any]] = []

    for row in rows:
        label = row["report_date"] or ""

        try:
            label = datetime.fromisoformat(label).strftime("%b")
        except ValueError:
            label = label[:10] if label else "Report"

        trend.append(
            {
                "month": label,
                "score": int(row["risk_score"] or 0),
            }
        )

    return trend


def _build_dashboard_trend(conn) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT report_date, risk_score
        FROM reports
        ORDER BY created_at ASC
        LIMIT 24
        """
    ).fetchall()

    if not rows:
        return [
            {
                "month": "Current",
                "score": 100,
            }
        ]

    buckets: dict[str, list[int]] = {}

    for row in rows:
        raw_date = str(row["report_date"] or "")

        try:
            bucket = datetime.fromisoformat(
                raw_date
            ).strftime("%b %Y")
        except ValueError:
            bucket = raw_date[:7] if raw_date else "Unknown"

        buckets.setdefault(bucket, []).append(
            int(row["risk_score"] or 0)
        )

    trend: List[Dict[str, Any]] = []

    for bucket, scores in buckets.items():
        trend.append(
            {
                "month": bucket,
                "score": int(
                    round(
                        sum(scores) / max(len(scores), 1)
                    )
                ),
            }
        )

    return trend[-6:]


def _format_activity_time(value: Any) -> str:
    if not value:
        return datetime.utcnow().strftime("%I:%M %p")

    try:
        return datetime.fromisoformat(
            str(value)
        ).strftime("%I:%M %p")
    except ValueError:
        return str(value)


def _build_dashboard_activity(conn) -> List[Dict[str, Any]]:
    report_rows = conn.execute(
        """
        SELECT id, report_type, status, created_at
        FROM reports
        ORDER BY created_at DESC
        LIMIT 4
        """
    ).fetchall()

    chat_rows = conn.execute(
        """
        SELECT id, role, content, created_at
        FROM chat_history
        ORDER BY created_at DESC
        LIMIT 4
        """
    ).fetchall()

    activities: list[
        tuple[datetime, dict[str, Any]]
    ] = []

    for row in report_rows:
        created_at = str(row["created_at"] or "")

        try:
            parsed = datetime.fromisoformat(created_at)
        except ValueError:
            parsed = datetime.utcnow()

        activities.append(
            (
                parsed,
                {
                    "id": f"report-{row['id']}",
                    "label": (
                        f"Report: "
                        f"{row['report_type'] or 'Uploaded Report'} "
                        f"({row['status'] or 'attention'})"
                    ),
                    "time": _format_activity_time(
                        created_at
                    ),
                },
            )
        )

    for row in chat_rows:
        created_at = str(row["created_at"] or "")

        try:
            parsed = datetime.fromisoformat(created_at)
        except ValueError:
            parsed = datetime.utcnow()

        snippet = (
            str(row["content"] or "")
            .strip()
            .replace("\n", " ")
        )

        if len(snippet) > 56:
            snippet = f"{snippet[:56].rstrip()}..."

        activities.append(
            (
                parsed,
                {
                    "id": f"chat-{row['id']}",
                    "label": (
                        f"{row['role'].title()} message: "
                        f"{snippet or 'No content'}"
                    ),
                    "time": _format_activity_time(
                        created_at
                    ),
                },
            )
        )

    activities.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        item[1]
        for item in activities[:6]
    ]


def _build_dashboard_summary(conn) -> Dict[str, Any]:
    report_rows = conn.execute(
        """
        SELECT
            id,
            status,
            risk_score,
            report_type,
            patient_name,
            hospital,
            report_date,
            total_parameters,
            abnormal_parameters,
            created_at
        FROM reports
        ORDER BY created_at DESC
        """
    ).fetchall()

    chat_count_row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM chat_history
        """
    ).fetchone()

    report_count = len(report_rows)

    high_risk_count = sum(
        1
        for row in report_rows
        if (
            (row["status"] or "") in {"critical", "high"}
            or float(row["risk_score"] or 0) >= 70
        )
    )

    normal_count = sum(
        1
        for row in report_rows
        if (row["status"] or "") == "normal"
    )

    conversation_count = int(
        chat_count_row["count"]
        if chat_count_row
        else 0
    )

    avg_risk = (
        sum(
            float(row["risk_score"] or 0)
            for row in report_rows
        )
        / max(report_count, 1)
    )

    health_score = max(
        0,
        min(
            100,
            int(
                round(
                    100
                    - avg_risk
                    - high_risk_count * 1.5
                )
            ),
        ),
    )

    recent_reports = [
        ReportSummaryResponse(
            id=row["id"],
            patient_name=row["patient_name"] or "Patient",
            hospital=row["hospital"] or "Unknown",
            report_date=(
                row["report_date"]
                or datetime.utcnow().isoformat()
            ),
            report_type=(
                row["report_type"]
                or "Uploaded Report"
            ),
            status=row["status"] or "attention",
            risk_score=int(row["risk_score"] or 0),
            total_parameters=int(
                row["total_parameters"] or 0
            ),
            abnormal_parameters=int(
                row["abnormal_parameters"] or 0
            ),
        )
        for row in report_rows[:5]
    ]

    return {
        "report_count": report_count,
        "high_risk_count": high_risk_count,
        "normal_count": normal_count,
        "conversation_count": conversation_count,
        "health_score": health_score,
        "trend": _build_dashboard_trend(conn),
        "recent_activity": _build_dashboard_activity(conn),
        "recent_reports": recent_reports,
    }


# ---------------------------------------------------------------------------
# Report context
# ---------------------------------------------------------------------------

def _resolve_report_context(
    conn,
    report_id: str | None,
) -> tuple[str, str | None]:
    if report_id:
        row = conn.execute(
            """
            SELECT id, raw_text
            FROM reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()

        if row:
            return (
                row["raw_text"] or "",
                row["id"],
            )

    row = conn.execute(
        """
        SELECT id, raw_text
        FROM reports
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()

    if row:
        return (
            row["raw_text"] or "",
            row["id"],
        )

    return "", None


# ---------------------------------------------------------------------------
# Render-safe startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup_bootstrap() -> None:
    # Models are trained during the Render build phase.
    # Do not train models or initialize the RAG index during
    # runtime startup because the free Render instance has
    # a 512 MiB memory limit.
    pass


# ---------------------------------------------------------------------------
# Report detail
# ---------------------------------------------------------------------------

def _load_report_detail(
    report_id: str,
) -> Dict[str, Any]:
    with get_connection() as conn:
        report_row = conn.execute(
            """
            SELECT *
            FROM reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()

        if not report_row:
            raise HTTPException(
                status_code=404,
                detail="Report not found",
            )

        parameter_rows = conn.execute(
            """
            SELECT
                id,
                name,
                value,
                unit,
                normal_range,
                status
            FROM extracted_parameters
            WHERE report_id = ?
            """,
            (report_id,),
        ).fetchall()

        prediction_rows = conn.execute(
            """
            SELECT
                id,
                name,
                probability,
                confidence,
                severity,
                summary,
                shap_summary
            FROM predictions
            WHERE report_id = ?
            """,
            (report_id,),
        ).fetchall()

        parameters = _normalize_parameter_rows(
            parameter_rows
        )

        if not parameters:
            fallback_parameters = _safe_json_loads(
                report_row["extracted_data"],
                [],
            )

            parameters = [
                {
                    "id": item.get("id")
                    or f"{report_id}-param-{index}",
                    "name": item.get("name", ""),
                    "value": item.get("value", 0),
                    "unit": item.get("unit", ""),
                    "normalRange": (
                        item.get("normalRange")
                        or item.get("normal_range")
                        or ""
                    ),
                    "status": item.get(
                        "status",
                        "normal",
                    ),
                }
                for index, item in enumerate(
                    fallback_parameters
                )
            ]

        predictions = _normalize_prediction_rows(
            prediction_rows
        )

        if not predictions:
            fallback_predictions = _safe_json_loads(
                report_row["prediction_summary"],
                [],
            )

            predictions = [
                _prediction_from_row(
                    item,
                    index=index,
                    report_id=report_id,
                )
                for index, item in enumerate(
                    fallback_predictions
                )
            ]

        explanation = explainability_service.explain(
            parameters,
            predictions,
            report_id=report_id,
        )

        risk_score = int(
            report_row["risk_score"] or 0
        )

        report_type = (
            report_row["report_type"]
            or "Uploaded Report"
        )

        patient_name = (
            report_row["patient_name"]
            or "Patient"
        )

        return {
            "id": report_row["id"],
            "patientName": patient_name,
            "hospital": (
                report_row["hospital"]
                or "Unknown"
            ),
            "reportDate": report_row["report_date"],
            "reportType": report_type,
            "status": (
                report_row["status"]
                or "attention"
            ),
            "riskScore": risk_score,
            "totalParameters": (
                report_row["total_parameters"]
                or 0
            ),
            "abnormalParameters": (
                report_row["abnormal_parameters"]
                or 0
            ),
            "parameters": parameters,
            "risks": predictions,
            "contributions": explanation[
                "contributions"
            ],
            "aiExplanation": (
                report_row["ai_explanation"]
                or explanation["explanation"]
            ),
            "lifestyleSuggestions": explanation[
                "lifestyle_suggestions"
            ],
            "recommendations": explanation[
                "recommendations"
            ],
            "shapValues": explanation[
                "shap_values"
            ],
            "topFeatures": explanation[
                "top_features"
            ],
            "featureImportanceVisualizationData": (
                explanation[
                    "feature_importance_visualization_data"
                ]
            ),
            "radarData": _build_radar_data(
                report_type,
                parameters,
                risk_score,
            ),
            "trend": _build_trend(
                conn,
                patient_name,
            ),
        }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_report(
    file: UploadFile = File(...),
) -> UploadResponse:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF uploads are supported",
        )

    UPLOADS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(file.filename).name

    file_path = (
        UPLOADS_DIR
        / f"{uuid.uuid4()}_{safe_filename}"
    )

    with file_path.open("wb") as buffer:
        content = await file.read()
        buffer.write(content)

    report_id = str(uuid.uuid4())[:8]

    text = (
        pdf_service.extract_text(
            str(file_path)
        )
        if file.filename.lower().endswith(".pdf")
        else ""
    )

    extraction_result = (
        extraction_service.extract_parameters(text)
    )

    parameters = extraction_result["parameters"]

    predictions = prediction_service.predict(
        parameters
    )

    explanation = explainability_service.explain(
        parameters,
        predictions,
        report_id=report_id,
    )

    report_status = _status_from_predictions(
        predictions
    )

    risk_score = int(
        max(
            (
                prediction["probability"]
                for prediction in predictions
            ),
            default=0,
        )
    )

    current_timestamp = datetime.utcnow().isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO reports (
                id,
                patient_name,
                hospital,
                report_date,
                report_type,
                status,
                risk_score,
                total_parameters,
                abnormal_parameters,
                raw_text,
                extracted_data,
                prediction_summary,
                ai_explanation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                "Patient",
                "Sample Hospital",
                current_timestamp,
                "Uploaded Report",
                report_status,
                risk_score,
                len(parameters),
                sum(
                    1
                    for param in parameters
                    if param.get("status")
                    != "normal"
                ),
                text,
                json.dumps(
                    parameters,
                    default=_json_default,
                ),
                json.dumps(
                    predictions,
                    default=_json_default,
                ),
                explanation["explanation"],
                current_timestamp,
            ),
        )

        for param in parameters:
            conn.execute(
                """
                INSERT INTO extracted_parameters(
                    report_id,
                    name,
                    value,
                    unit,
                    normal_range,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    param["name"],
                    param["value"],
                    param.get("unit", ""),
                    (
                        param.get("normal_range", "")
                        or param.get(
                            "normalRange",
                            "",
                        )
                    ),
                    param.get(
                        "status",
                        "normal",
                    ),
                ),
            )

        for pred in predictions:
            conn.execute(
                """
                INSERT INTO predictions(
                    report_id,
                    name,
                    probability,
                    confidence,
                    severity,
                    summary,
                    shap_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    pred["name"],
                    pred["probability"],
                    pred["confidence"],
                    pred["severity"],
                    pred["summary"],
                    json.dumps(
                        pred,
                        default=_json_default,
                    ),
                ),
            )

        conn.commit()

    return UploadResponse(
        report_id=report_id,
        message="Report uploaded successfully",
        extracted_text=text,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@app.post(
    "/extract",
    response_model=ExtractResponse,
)
async def extract(
    payload: ExtractRequest,
) -> ExtractResponse:
    result = extraction_service.extract_parameters(
        payload.text
    )

    return ExtractResponse(
        parameters=result["parameters"]
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    payload: PredictRequest,
) -> PredictResponse:
    predictions = prediction_service.predict(
        payload.parameters
    )

    return PredictResponse(
        predictions=predictions
    )


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

@app.post(
    "/explain",
    response_model=ExplainResponse,
)
async def explain(
    payload: ExplainRequest,
) -> ExplainResponse:

    explanation = explainability_service.explain(
        payload.parameters,
        payload.predictions,
        report_id=payload.report_id,
    )

    return ExplainResponse(
        explanation=explanation["explanation"],
        lifestyle_suggestions=explanation[
            "lifestyle_suggestions"
        ],
        recommendations=explanation[
            "recommendations"
        ],
        shap_values=explanation["shap_values"],
        top_features=explanation["top_features"],
        feature_importance_visualization_data=(
            explanation[
                "feature_importance_visualization_data"
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
) -> ChatResponse:

    with get_connection() as conn:
        report_context, resolved_report_id = (
            _resolve_report_context(
                conn,
                payload.report_id,
            )
        )

        response = rag_service.answer(
            payload.message,
            report_context,
            report_id=resolved_report_id,
        )

        conn.execute(
            """
            INSERT INTO chat_history(
                report_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                resolved_report_id,
                "user",
                payload.message,
                datetime.utcnow().isoformat(),
            ),
        )

        conn.execute(
            """
            INSERT INTO chat_history(
                report_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                resolved_report_id,
                "assistant",
                response,
                datetime.utcnow().isoformat(),
            ),
        )

        conn.commit()

    return ChatResponse(
        response=response
    )


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
):
    with get_connection() as conn:
        report_context, resolved_report_id = (
            _resolve_report_context(
                conn,
                payload.report_id,
            )
        )

    def _generator():
        chunks: list[str] = []

        for chunk in rag_service.stream(
            payload.message,
            report_context,
            report_id=resolved_report_id,
        ):
            if chunk:
                chunks.append(chunk)
                yield chunk

        response_text = "".join(chunks).strip()

        with get_connection() as conn:
            timestamp = datetime.utcnow().isoformat()

            conn.execute(
                """
                INSERT INTO chat_history(
                    report_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_report_id,
                    "user",
                    payload.message,
                    timestamp,
                ),
            )

            conn.execute(
                """
                INSERT INTO chat_history(
                    report_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_report_id,
                    "assistant",
                    response_text,
                    timestamp,
                ),
            )

            conn.commit()

    return StreamingResponse(
        _generator(),
        media_type="text/plain; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

@app.get(
    "/chat/history",
    response_model=ChatHistoryResponse,
)
def chat_history(
    report_id: str | None = None,
    report_context: str = "",
) -> ChatHistoryResponse:

    payload = rag_service.history_payload(
        report_id=report_id,
        report_context=report_context,
    )

    return ChatHistoryResponse(**payload)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
def dashboard_summary() -> DashboardSummaryResponse:

    with get_connection() as conn:
        payload = _build_dashboard_summary(conn)

    return DashboardSummaryResponse(**payload)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.get(
    "/reports",
    response_model=List[ReportSummaryResponse],
)
async def list_reports() -> List[ReportSummaryResponse]:

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                patient_name,
                hospital,
                report_date,
                report_type,
                status,
                risk_score,
                total_parameters,
                abnormal_parameters
            FROM reports
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [
        ReportSummaryResponse(
            id=row["id"],
            patient_name=(
                row["patient_name"]
                or "Patient"
            ),
            hospital=(
                row["hospital"]
                or "Unknown"
            ),
            report_date=row["report_date"],
            report_type=(
                row["report_type"]
                or "Uploaded Report"
            ),
            status=(
                row["status"]
                or "attention"
            ),
            risk_score=row["risk_score"] or 0,
            total_parameters=(
                row["total_parameters"]
                or 0
            ),
            abnormal_parameters=(
                row["abnormal_parameters"]
                or 0
            ),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Individual report
# ---------------------------------------------------------------------------

@app.get("/report/{report_id}")
async def get_report(
    report_id: str,
) -> Dict[str, Any]:
    return _load_report_detail(report_id)


# ---------------------------------------------------------------------------
# Download report
# ---------------------------------------------------------------------------

@app.get("/report/{report_id}/download")
async def download_report(
    report_id: str,
) -> Response:

    report = _load_report_detail(report_id)

    pdf_bytes = pdf_service.build_report_pdf(
        report
    )

    filename = (
        f"medilens-report-{report_id}.pdf"
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        },
    )


# ---------------------------------------------------------------------------
# Delete report
# ---------------------------------------------------------------------------

@app.delete("/report/{report_id}")
async def delete_report(
    report_id: str,
) -> Dict[str, str]:

    with get_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM reports
            WHERE id = ?
            """,
            (report_id,),
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Report not found",
            )

        conn.commit()

    return {"status": "deleted"}
