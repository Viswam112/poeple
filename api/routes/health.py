import time
import torch
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from database.connection import get_db

router = APIRouter(tags=["Health & Metrics"])

pipeline_workers_ref = {}

def register_workers_health(workers_dict):
    global pipeline_workers_ref
    pipeline_workers_ref = workers_dict

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    cameras_status = {}
    for cid, worker in pipeline_workers_ref.items():
        cameras_status[cid] = "ONLINE" if worker.reader.is_connected else "OFFLINE"

    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database": "connected" if db_ok else "disconnected",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cameras": cameras_status,
        "timestamp": time.time()
    }

@router.get("/metrics")
def get_metrics():
    metrics = []
    for cid, worker in pipeline_workers_ref.items():
        metrics.append({
            "camera_id": cid,
            "camera_fps": worker.reader.fps,
            "processing_fps": worker.processing_fps,
            "detection_latency_ms": round(worker.detection_latency_ms, 2),
            "tracking_latency_ms": round(worker.tracking_latency_ms, 2),
            "end_to_end_latency_ms": round(worker.end_to_end_latency_ms, 2),
            "status": "ONLINE" if worker.reader.is_connected else "OFFLINE"
        })
    return {"metrics": metrics}
