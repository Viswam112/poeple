import time
from typing import Dict, Any
from fastapi import APIRouter

from database.connection import SessionLocal
from database.repository import EventRepository

router = APIRouter(prefix="/occupancy", tags=["Occupancy"])

# In-memory worker statistics registry reference set by main pipeline
pipeline_workers_ref = {}

# Peak of the building-wide total (not the max of any single camera)
_peak_total_occupancy = 0

# Today's usage is replayed from the events table; the dashboard polls every second, so cache it
_usage_cache: Dict[str, Any] = {}
_usage_cached_at: float = 0.0
USAGE_CACHE_SECONDS = 20.0


def _today_usage() -> Dict[str, Any]:
    global _usage_cache, _usage_cached_at
    now = time.time()
    if _usage_cache and now - _usage_cached_at < USAGE_CACHE_SECONDS:
        return _usage_cache
    try:
        db = SessionLocal()
        try:
            _usage_cache = EventRepository.get_today_usage(db)
        finally:
            db.close()
        _usage_cached_at = now
    except Exception:
        _usage_cache = _usage_cache or {"average_occupancy": 0.0, "peak_occupancy": 0, "peak_time": None, "since": None}
    return _usage_cache


def register_workers(workers_dict):
    global pipeline_workers_ref
    pipeline_workers_ref = workers_dict


def _camera_stats(cid: str, worker) -> Dict[str, Any]:
    stats = worker.counting_engine.get_stats()
    stats["name"] = worker.camera_name
    stats["status"] = "ONLINE" if worker.reader.is_connected else "OFFLINE"
    stats["processing_fps"] = worker.processing_fps
    stats["end_to_end_latency_ms"] = round(worker.end_to_end_latency_ms, 1)
    stats["last_processed_at"] = worker.last_processed_at
    stats["frame_width"] = worker.frame_width
    stats["frame_height"] = worker.frame_height
    stats["exclude_zone_count"] = len(worker.exclusion.polygons)
    return stats


def get_summary() -> Dict[str, Any]:
    global _peak_total_occupancy
    total_curr = total_ent = total_ext = 0
    cam_stats = []

    for cid, worker in list(pipeline_workers_ref.items()):
        stats = _camera_stats(cid, worker)
        total_curr += stats["current_occupancy"]
        total_ent += stats["total_entries"]
        total_ext += stats["total_exits"]
        cam_stats.append(stats)

    _peak_total_occupancy = max(_peak_total_occupancy, total_curr)
    cam_stats.sort(key=lambda s: s["camera_id"])
    usage = _today_usage()

    return {
        "current_occupancy": total_curr,
        "total_entries": total_ent,
        "total_exits": total_ext,
        "peak_occupancy": max(_peak_total_occupancy, usage.get("peak_occupancy", 0)),
        "average_occupancy": usage.get("average_occupancy", 0.0),
        "peak_time": usage.get("peak_time"),
        "tracking_since": usage.get("since"),
        "cameras_online": sum(1 for s in cam_stats if s["status"] == "ONLINE"),
        "cameras_total": len(cam_stats),
        "cameras": cam_stats,
        "timestamp": time.time(),
    }


@router.get("")
def get_total_occupancy():
    return get_summary()


@router.get("/{camera_id}")
def get_camera_occupancy(camera_id: str):
    if camera_id in pipeline_workers_ref:
        return _camera_stats(camera_id, pipeline_workers_ref[camera_id])
    return {"camera_id": camera_id, "current_occupancy": 0, "total_entries": 0, "total_exits": 0, "peak_occupancy": 0}
