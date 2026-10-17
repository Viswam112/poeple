import os
import json
import time
import queue
import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Column, Float, Index, Integer, String, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from ultralytics import YOLO
from dongle_controller import DongleController
from light_automation import LightAutomation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


# ============================================================================
# Settings
# ============================================================================

class Settings:
    PROJECT_NAME: str = "Workspace People Counting System"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Computer Vision defaults
    # yolo11m over yolov8m: same speed on this CPU (~165 ms/frame) but it does not mistake
    # high-backed office chairs for people, which yolov8m did at up to 0.66 confidence.
    MODEL_PATH: str = os.getenv("MODEL_PATH", "yolo11m_openvino_model")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")  # "cuda", "cpu", or "auto"
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "640"))

    # Two confidence levels (ByteTrack-style):
    #   DETECTION_FLOOR      - detector keeps everything above this; used only to KEEP an existing track alive
    #   CONFIDENCE_THRESHOLD - a detection must reach this to START a new track (per-camera override in cameras.json)
    # Concurrent inferences. A single inference leaves most of this CPU idle, so serialising every
    # camera through one model instance was capping the whole system at ~2.9 detections/second.
    # Measured on this machine: 1 -> 2.9/s, 2 -> ~4.1/s, 4 -> ~8/s. Each instance costs ~40 MB.
    DETECTOR_POOL_SIZE: int = int(os.getenv("DETECTOR_POOL_SIZE", "4"))
    DETECTION_FLOOR: float = float(os.getenv("DETECTION_FLOOR", "0.20"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

    # Tracking & Counting - all time based so behaviour is the same at 1 FPS (10 cameras) or 12 FPS (1 camera)
    TRACK_CONFIRM_SECONDS: float = float(os.getenv("TRACK_CONFIRM_SECONDS", "2.0"))       # min age before a track counts
    TRACK_CONFIRM_MIN_HITS: int = int(os.getenv("TRACK_CONFIRM_MIN_HITS", "5"))           # min detections before it counts
    TRACK_CONFIRM_HIT_RATIO: float = float(os.getenv("TRACK_CONFIRM_HIT_RATIO", "0.75"))   # detected in >=75% of frames since created
    TRACK_CONFIRM_MEAN_CONF: float = float(os.getenv("TRACK_CONFIRM_MEAN_CONF", "0.30"))  # average confidence must be at least this
    # The detector must have been convinced at least ONCE before we count something as a person.
    # An empty office chair is detected weakly but persistently: measured over 3 minutes on the live
    # cameras, chair tracks peaked at 0.44 and 0.46 while the weakest real person peaked at 0.55.
    # Neither the average confidence nor the chair class separates them (the model does not even
    # label those chairs as chairs), but the peak does.
    TRACK_CONFIRM_PEAK_CONF: float = float(os.getenv("TRACK_CONFIRM_PEAK_CONF", "0.50"))
    TRACK_LOST_SECONDS: float = float(os.getenv("TRACK_LOST_SECONDS", "6.0"))             # unseen this long -> EXIT
    MAX_PLAUSIBLE_OCCUPANCY: int = int(os.getenv("MAX_PLAUSIBLE_OCCUPANCY", "50"))       # per camera; guards DB restore
    REFERENCE_POINT: str = os.getenv("REFERENCE_POINT", "TOP_CENTER")
    # Per-camera processing rate. Occupancy of seated people does not need a high rate, and this CPU
    # runs one detection in ~180 ms, so all cameras together can do ~5 per second. Asking for more
    # just makes every camera queue behind the others and inflates the reported latency.
    TARGET_PROCESSING_FPS: int = int(os.getenv("TARGET_PROCESSING_FPS", "3"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./people_counter.db")

    # Config file path for persistent camera setup
    CAMERAS_CONFIG_PATH: str = os.getenv("CAMERAS_CONFIG_PATH", "config/cameras.json")

    # Lighting stays disabled until local bridge commands are configured and enabled by an operator.
    LIGHT_AUTO_ENABLED: bool = os.getenv("LIGHT_AUTO_ENABLED", "false").lower() in {"1", "true", "yes"}
    LIGHT_AUTO_OFF_DELAY_SECONDS: int = int(os.getenv("LIGHT_AUTO_OFF_DELAY_SECONDS", "300"))
    LIGHT_COMMAND_RETRY_COUNT: int = int(os.getenv("LIGHT_COMMAND_RETRY_COUNT", "3"))
    LIGHT_COMMAND_RETRY_DELAY: float = float(os.getenv("LIGHT_COMMAND_RETRY_DELAY", "1"))
    LIGHT_COMMAND_TIMEOUT_SECONDS: float = float(os.getenv("LIGHT_COMMAND_TIMEOUT_SECONDS", "10"))
    LIGHT_ON_COMMAND: str = os.getenv("LIGHT_ON_COMMAND", "")
    LIGHT_OFF_COMMAND: str = os.getenv("LIGHT_OFF_COMMAND", "")


settings = Settings()
os.makedirs("config", exist_ok=True)

dongle_controller = DongleController(
    on_command=settings.LIGHT_ON_COMMAND,
    off_command=settings.LIGHT_OFF_COMMAND,
    retry_count=settings.LIGHT_COMMAND_RETRY_COUNT,
    retry_delay=settings.LIGHT_COMMAND_RETRY_DELAY,
    timeout=settings.LIGHT_COMMAND_TIMEOUT_SECONDS,
)
light_automation = LightAutomation(
    dongle=dongle_controller,
    enabled=settings.LIGHT_AUTO_ENABLED,
    off_delay_seconds=settings.LIGHT_AUTO_OFF_DELAY_SECONDS,
)


# ============================================================================
# Database
# ============================================================================

Base = declarative_base()


class TrackEventModel(Base):
    """
    One ENTRY or EXIT for one tracked person.

    This is the only table the system keeps: occupancy, the daily totals, the average and the peak
    are all reconstructed by replaying these events, so nothing else needs to be stored.
    """
    __tablename__ = "track_events"

    event_id = Column(String(100), primary_key=True, index=True)
    camera_id = Column(String(50), index=True, nullable=False)
    track_id = Column(Integer, nullable=False)
    event_type = Column(String(20), nullable=False)  # ENTRY or EXIT
    zone = Column(String(50), default="workspace")
    timestamp = Column(Float, index=True, nullable=False)
    confidence = Column(Float, default=0.0)


Index("idx_track_events_cam_time", TrackEventModel.camera_id, TrackEventModel.timestamp)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class EventRepository:
    @staticmethod
    def insert_event_idempotent(db: Session, event_data: Dict[str, Any]) -> bool:
        event_id = event_data["event_id"]
        existing = db.query(TrackEventModel).filter(TrackEventModel.event_id == event_id).first()
        if existing:
            return False  # Already processed

        track_event = TrackEventModel(
            event_id=event_id,
            camera_id=event_data["camera_id"],
            track_id=event_data["track_id"],
            event_type=event_data["event_type"],
            timestamp=event_data["timestamp"],
            confidence=event_data.get("confidence", 0.0)
        )
        db.add(track_event)
        db.commit()
        return True

    @staticmethod
    def close_orphan_sessions(db: Session, camera_id: str, day_start: Optional[float] = None) -> int:
        """
        Balance today's event log for one camera by closing sessions that were never closed.

        At startup nobody is being tracked, so today's ENTRY and EXIT counts for this camera must
        match. If they do not, a previous run ended without closing its tracks - a force-kill, a
        crash, or a power cut - and the log claims those people are still present. Left alone that
        inflates the daily average and peak for the rest of the day, and it compounds with every
        restart. Writes the missing EXIT events, timed at the camera's last known event.
        """
        if day_start is None:
            day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        rows = (db.query(TrackEventModel.timestamp, TrackEventModel.event_type)
                  .filter(TrackEventModel.camera_id == camera_id, TrackEventModel.timestamp >= day_start)
                  .order_by(TrackEventModel.timestamp.asc()).all())
        if not rows:
            return 0
        entries = sum(1 for _, k in rows if k == "ENTRY")
        exits = sum(1 for _, k in rows if k == "EXIT")
        orphans = entries - exits
        if orphans <= 0:
            return 0

        last_ts = rows[-1][0]
        for i in range(orphans):
            EventRepository.insert_event_idempotent(db, {
                "event_id": f"{camera_id}_orphan{i}_EXIT_{int(last_ts)}",
                "camera_id": camera_id,
                "track_id": -1,
                "event_type": "EXIT",
                "timestamp": last_ts,
                "confidence": 0.0,
                "current_occupancy": 0,
            })
        return orphans

    @staticmethod
    def get_today_usage(db: Session, day_start: Optional[float] = None) -> Dict[str, Any]:
        """
        Replay today's ENTRY/EXIT events across all cameras into a building-wide occupancy timeline.

        Returns the time-weighted average occupancy (how busy the space has been, not just how busy
        it is right now), the peak and the clock time the peak happened. The average is taken from
        the first event of the day rather than from midnight, so hours when the office was shut do
        not drag it to zero.
        """
        if day_start is None:
            day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        rows = (db.query(TrackEventModel.timestamp, TrackEventModel.event_type)
                  .filter(TrackEventModel.timestamp >= day_start)
                  .order_by(TrackEventModel.timestamp.asc()).all())
        if not rows:
            return {"average_occupancy": 0.0, "peak_occupancy": 0, "peak_time": None, "since": None}

        now = time.time()
        occ = peak = 0
        peak_time = rows[0][0]
        first_ts = rows[0][0]
        prev_ts = first_ts
        weighted = 0.0
        for ts, kind in rows:
            weighted += occ * max(0.0, ts - prev_ts)      # occupancy held over the interval just ended
            prev_ts = ts
            occ = occ + 1 if kind == "ENTRY" else max(0, occ - 1)
            if occ > peak:
                peak, peak_time = occ, ts
        weighted += occ * max(0.0, now - prev_ts)          # up to now at the current level

        span = max(1.0, now - first_ts)
        return {
            "average_occupancy": round(weighted / span, 1),
            "peak_occupancy": peak,
            "peak_time": peak_time,
            "since": first_ts,
        }

    @staticmethod
    def get_daily_totals(db: Session, camera_id: str, day_start: Optional[float] = None) -> Dict[str, Any]:
        """Entries, exits and peak occupancy for one camera since local midnight, replayed from events."""
        if day_start is None:
            day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        rows = (db.query(TrackEventModel)
                  .filter(TrackEventModel.camera_id == camera_id, TrackEventModel.timestamp >= day_start)
                  .order_by(TrackEventModel.timestamp.asc()).all())
        entries = exits = occ = peak = 0
        for r in rows:
            if r.event_type == "ENTRY":
                entries += 1; occ += 1
            elif r.event_type == "EXIT":
                exits += 1; occ = max(0, occ - 1)
            peak = max(peak, occ)
        return {"entries": entries, "exits": exits, "peak": peak}


# ============================================================================
# Detection
# ============================================================================

class Detection:
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, class_id: int, class_name: str):
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class PersonDetector(ABC):
    """
    Abstract base class for Person Detection models.
    Allows replacing YOLO with TensorRT, ONNX, OpenVINO, etc.
    without breaking tracking or business logic.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Process a single image/frame and return a list of Detection objects
        strictly representing class 'person'.
        """
        pass

    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        pass


class YOLOPersonDetector(PersonDetector):
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.45, iou_threshold: float = 0.45, device: str = "auto", img_size: int = 640):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Exported backends (OpenVINO/ONNX dirs or files) manage their own device and reject .to()
        self.is_pytorch_model = str(self.model_path).endswith(".pt")
        self.model = YOLO(self.model_path, task="detect")
        if self.is_pytorch_model:
            self.model.to(self.device)

        # One detector instance is shared by every camera worker. The OpenVINO/torch inference
        # request is not re-entrant, so calls are serialised here.
        self._infer_lock = threading.Lock()

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold

    def detect(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        conf_thr = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        with self._infer_lock:
            results = self.model(
                frame,
                conf=conf_thr,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                cls_name = self.model.names[cls_id]

                # Strict filtering: CLASS MUST BE PERSON ONLY
                if cls_name == "person":
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0])

                    box_w = max(1.0, x2 - x1)
                    box_h = max(1.0, y2 - y1)
                    aspect_ratio = box_h / box_w

                    # Empty office chairs (tall, narrow boxes) can register as low-confidence "person".
                    # yolov8m scores chairs < 0.15 in testing while real seated people behind chair backs
                    # land around 0.40, so the guard sits between them. (Was 0.48 for the nano model.)
                    if conf < 0.30 and aspect_ratio > 1.7:
                        continue

                    # An office chair backrest seen edge-on from a ceiling camera is a very tall, narrow
                    # box that yolov8m can score as high as ~0.7 "person". Seated people from this angle
                    # are ~1.0-2.0; even a standing person foreshortened by the overhead view stays
                    # under ~2.6. Anything beyond that needs high confidence to pass (chair scored 0.68, standing person 0.81).
                    if aspect_ratio > 2.6 and conf < 0.75:
                        continue

                    # Reject small non-human items
                    if box_w < 35 or box_h < 45:
                        continue

                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="person"
                    ))

        return self._suppress_nested(detections)

    @staticmethod
    def _suppress_nested(detections: List[Detection], containment_threshold: float = 0.7) -> List[Detection]:
        """
        Standard NMS only compares IoU, so a small "head + shoulders" box sitting fully inside a
        larger full-body box of the same person can survive (IoU ~0.4). Drop any box whose area is
        mostly contained within a higher-confidence box.
        """
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = max(1.0, (x2 - x1) * (y2 - y1))
            nested = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k.bbox
                inter = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(0.0, min(y2, ky2) - max(y1, ky1))
                if inter / area > containment_threshold:
                    nested = True
                    break
            if not nested:
                kept.append(det)
        return kept


class DetectorPool:
    """
    A small pool of detector instances shared by every camera worker.

    One detector behind a lock was the previous design: it kept memory down, but a single OpenVINO
    inference only occupies about 2 of this machine's 12 cores, so serialising every camera through
    it capped the whole system at ~2.8 detections/second no matter how many cameras were running,
    and each camera's latency grew linearly with the number of cameras (378 ms with one camera,
    693 ms with two, 1087 ms with three).

    Running a couple of inferences concurrently uses the idle cores and roughly doubles aggregate
    throughput (~5.7/s measured). Throughput saturates at 2-3 concurrent inferences on this CPU, so
    a handful of instances serves any number of cameras - memory stays bounded as cameras are added.
    """

    def __init__(self, size: int = 2, **detector_kwargs):
        self.size = max(1, int(size))
        self._free: "queue.Queue[YOLOPersonDetector]" = queue.Queue()
        self._all: List[YOLOPersonDetector] = []
        for i in range(self.size):
            det = YOLOPersonDetector(**detector_kwargs)
            self._all.append(det)
            self._free.put(det)
        logger.info(f"Detector pool ready: {self.size} instance(s) of {detector_kwargs.get('model_path')}")

    def detect(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Detection]:
        """Borrow an instance for the duration of one inference, then return it to the pool."""
        det = self._free.get()
        try:
            return det.detect(frame, confidence_threshold=confidence_threshold)
        finally:
            self._free.put(det)

    def set_confidence_threshold(self, threshold: float) -> None:
        for det in self._all:
            det.set_confidence_threshold(threshold)


def build_detector():
    """The shared detector (pool) every camera worker uses."""
    return DetectorPool(
        size=settings.DETECTOR_POOL_SIZE,
        model_path=settings.MODEL_PATH,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        iou_threshold=settings.IOU_THRESHOLD,
        device=settings.MODEL_DEVICE,
        img_size=settings.IMAGE_SIZE,
    )


# ============================================================================
# Tracking
# ============================================================================

class Track:
    """
    One tracked person.

    Confirmation is evidence based and measured in seconds, not frames, so behaviour does not change
    with processing rate (one camera at 12 FPS vs ten cameras at ~1 FPS each):

      * a track becomes "confirmed" (counted) once it has existed for `confirm_seconds`, has at
        least `confirm_min_hits` detections, was detected in at least `confirm_hit_ratio` of the
        frames since it was created, and its average confidence over that time is at least
        `confirm_mean_conf`.  A bag or chair whose score only occasionally spikes over the line
        never satisfies the ratio / mean-confidence tests; a real person does within a second or two.
      * a track is "lost" when it has not been seen for `lost_seconds`.
    """

    # Never judge a track on fewer samples than this (at 5 FPS that is 2 s; at 1 FPS with ten cameras, 10 s)
    MIN_SAMPLES_TO_JUDGE = 10

    def __init__(self, track_id: int, initial_bbox: Tuple[float, float, float, float], confidence: float,
                 max_history: int = 30, now: Optional[float] = None):
        now = time.time() if now is None else now
        self.track_id = track_id
        self.bbox = initial_bbox
        self.confidence = confidence
        self.class_name = "person"

        self.first_seen = now
        self.last_seen = now
        self.frames_seen = 1          # detections matched to this track
        self.frames_total = 1         # frames this track has existed for (hits + misses)
        self.consecutive_misses = 0
        self.conf_sum = confidence
        self.peak_confidence = confidence
        self.velocity = (0.0, 0.0)    # px/s of the box centre, EMA-smoothed

        self.status = "tentative"     # tentative -> confirmed -> lost
        self.lost_reason = ""
        self.confirmed_at: Optional[float] = None
        self.zone = "unknown"

        self.trajectory = deque(maxlen=max_history)
        self._update_reference_points(initial_bbox)

    # ------------------------------------------------------------------ geometry
    def _update_reference_points(self, bbox: Tuple[float, float, float, float], reference_mode: str = "TOP_CENTER"):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if reference_mode == "TOP_CENTER" or reference_mode == "HEAD":
            ref_pt = (cx, y1)
        elif reference_mode == "BOTTOM_CENTER":
            ref_pt = (cx, y2)
        else:
            ref_pt = (cx, cy)
        self.trajectory.append(ref_pt)

    @property
    def current_point(self) -> Tuple[float, float]:
        if len(self.trajectory) > 0:
            return self.trajectory[-1]
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y1)

    def predicted_bbox(self, now: float, max_horizon: float = 1.0) -> Tuple[float, float, float, float]:
        """Where the box should be now, assuming constant velocity (capped so a stale track doesn't fly away)."""
        dt = min(max(0.0, now - self.last_seen), max_horizon)
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    # ------------------------------------------------------------------ evidence
    @property
    def hit_ratio(self) -> float:
        return self.frames_seen / float(max(1, self.frames_total))

    @property
    def mean_confidence(self) -> float:
        return self.conf_sum / float(max(1, self.frames_seen))

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen

    def update(self, bbox: Tuple[float, float, float, float], confidence: float, reference_mode: str = "TOP_CENTER",
               now: Optional[float] = None, confirm_seconds: float = 2.0, confirm_min_hits: int = 5,
               confirm_hit_ratio: float = 0.75, confirm_mean_conf: float = 0.30,
               confirm_peak_conf: float = 0.50):
        now = time.time() if now is None else now
        dt = now - self.last_seen
        if dt > 0:
            ocx, ocy = (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0
            ncx, ncy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            vx, vy = (ncx - ocx) / dt, (ncy - ocy) / dt
            a = 0.5
            self.velocity = (a * vx + (1 - a) * self.velocity[0], a * vy + (1 - a) * self.velocity[1])
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = now
        self.frames_seen += 1
        self.frames_total += 1
        self.consecutive_misses = 0
        self.conf_sum += confidence
        self.peak_confidence = max(self.peak_confidence, confidence)
        self._update_reference_points(bbox, reference_mode)

        if self.status == "tentative":
            ok = (self.age_seconds >= confirm_seconds
                  and self.frames_seen >= confirm_min_hits
                  and self.frames_total >= self.MIN_SAMPLES_TO_JUDGE
                  and self.hit_ratio >= confirm_hit_ratio
                  and self.mean_confidence >= confirm_mean_conf
                  and self.peak_confidence >= confirm_peak_conf)
            if ok:
                self.status = "confirmed"
                self.confirmed_at = now

    def mark_missed(self, now: Optional[float] = None, lost_seconds: float = 6.0):
        now = time.time() if now is None else now
        self.consecutive_misses += 1
        self.frames_total += 1
        self.velocity = (self.velocity[0] * 0.5, self.velocity[1] * 0.5)
        if now - self.last_seen > lost_seconds:
            self.status = "lost"
            self.lost_reason = "timeout"
        # A tentative track that keeps missing is dropped quickly: it never earned a place.
        # (Real people are detected in nearly every frame; flickering objects are not.)
        if self.status == "tentative" and self.frames_total >= 4 and self.hit_ratio < 0.5:
            self.status = "lost"


def iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return inter_area / float(area1 + area2 - inter_area)


def containment(bbox1, bbox2) -> float:
    """Intersection over the smaller box's area. Robust to a box flipping between head-only and full-body."""
    x1 = max(bbox1[0], bbox2[0]); y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2]); y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter_area / float(max(1.0, min(area1, area2)))


def area_ratio(bbox1, bbox2) -> float:
    a1 = max(1.0, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    a2 = max(1.0, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
    return max(a1, a2) / min(a1, a2)


def match_score(bbox1, bbox2, max_area_ratio: float = 3.0) -> float:
    """
    Overlap score for association. Containment lets a box that flips between head-only and full
    body stay on one track, but a small box inside a much larger one (seated person behind a
    standing one) is a different person, so wildly different sizes score by IoU only.
    """
    if area_ratio(bbox1, bbox2) > max_area_ratio:
        return iou(bbox1, bbox2)
    return max(iou(bbox1, bbox2), containment(bbox1, bbox2))


class MultiObjectTracker:
    """
    IoU/containment tracker with ByteTrack-style two-stage association.

    * Detections at or above `new_track_confidence` may start a new track.
    * Detections down to the detector floor may only *continue* an existing track.  A seated person
      whose score sags to 0.25 for a while therefore keeps their identity, while a bag that flickers
      to 0.32 for a few frames never becomes a track at all.
    * Track confirmation and loss are evidence- and time-based (see Track).
    * Overlapping duplicate tracks on one person are merged.

    `update()` returns the tracks currently counted (confirmed and not lost).  Tracks that were
    confirmed and have just been dropped are available in `exited_tracks` until the next update.
    """

    def __init__(self, new_track_confidence: float = 0.35, match_threshold: float = 0.3,
                 confirm_seconds: float = 2.0, confirm_min_hits: int = 5, confirm_hit_ratio: float = 0.75,
                 confirm_mean_conf: float = 0.30, confirm_peak_conf: float = 0.50, lost_seconds: float = 6.0,
                 reference_mode: str = "TOP_CENTER", duplicate_overlap: float = 0.6):
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.exited_tracks: List[Track] = []

        self.new_track_confidence = new_track_confidence
        self.match_threshold = match_threshold
        self.confirm_seconds = confirm_seconds
        self.confirm_min_hits = confirm_min_hits
        self.confirm_hit_ratio = confirm_hit_ratio
        self.confirm_mean_conf = confirm_mean_conf
        self.confirm_peak_conf = confirm_peak_conf
        self.lost_seconds = lost_seconds
        self.reference_mode = reference_mode
        self.duplicate_overlap = duplicate_overlap

    # ------------------------------------------------------------------ association
    def _associate(self, track_ids: List[int], det_indices: List[int], detections: List[Detection],
                   now: float) -> Tuple[set, set]:
        """Greedy best-overlap matching between the given tracks and detections."""
        matched_tracks, matched_dets = set(), set()
        if not track_ids or not det_indices:
            return matched_tracks, matched_dets

        cost = np.zeros((len(track_ids), len(det_indices)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            for j, di in enumerate(det_indices):
                cost[i, j] = match_score(self.tracks[tid].predicted_bbox(now), detections[di].bbox)

        while cost.size:
            i, j = np.unravel_index(np.argmax(cost), cost.shape)
            if cost[i, j] < self.match_threshold:
                break
            tid, di = track_ids[i], det_indices[j]
            det = detections[di]
            self.tracks[tid].update(
                det.bbox, det.confidence, reference_mode=self.reference_mode, now=now,
                confirm_seconds=self.confirm_seconds, confirm_min_hits=self.confirm_min_hits,
                confirm_hit_ratio=self.confirm_hit_ratio, confirm_mean_conf=self.confirm_mean_conf,
                confirm_peak_conf=self.confirm_peak_conf,
            )
            matched_tracks.add(tid)
            matched_dets.add(di)
            cost[i, :] = -1.0
            cost[:, j] = -1.0
        return matched_tracks, matched_dets

    def _associate_by_distance(self, track_ids: List[int], det_indices: List[int], detections: List[Detection],
                               now: float, max_dist_factor: float = 1.0) -> Tuple[set, set]:
        """
        Fallback for people who moved too far for boxes to overlap (walking between frames, or a
        track that was missed for a frame or two): match by centre distance, gated to one box
        diagonal of the predicted position. Prevents an EXIT + ENTRY pair for a single person.
        """
        matched_tracks, matched_dets = set(), set()
        if not track_ids or not det_indices:
            return matched_tracks, matched_dets
        pairs = []
        for tid in track_ids:
            px1, py1, px2, py2 = self.tracks[tid].predicted_bbox(now)
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            diag = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
            for di in det_indices:
                x1, y1, x2, y2 = detections[di].bbox
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                # boxes must also be of comparable size (a far-away small box is a different person)
                ar = ((x2 - x1) * (y2 - y1)) / max(1.0, (px2 - px1) * (py2 - py1))
                if dist <= max_dist_factor * diag and 0.4 <= ar <= 2.5:
                    pairs.append((dist / max(1.0, diag), tid, di))
        for _, tid, di in sorted(pairs):
            if tid in matched_tracks or di in matched_dets:
                continue
            det = detections[di]
            self.tracks[tid].update(
                det.bbox, det.confidence, reference_mode=self.reference_mode, now=now,
                confirm_seconds=self.confirm_seconds, confirm_min_hits=self.confirm_min_hits,
                confirm_hit_ratio=self.confirm_hit_ratio, confirm_mean_conf=self.confirm_mean_conf,
                confirm_peak_conf=self.confirm_peak_conf,
            )
            matched_tracks.add(tid)
            matched_dets.add(di)
        return matched_tracks, matched_dets

    def update(self, detections: List[Detection], now: Optional[float] = None) -> List[Track]:
        now = time.time() if now is None else now
        self.exited_tracks = []

        high = [i for i, d in enumerate(detections) if d.confidence >= self.new_track_confidence]
        low = [i for i, d in enumerate(detections) if d.confidence < self.new_track_confidence]

        # Stage 1: all tracks vs high-confidence detections
        track_ids = list(self.tracks.keys())
        m_tracks, m_dets = self._associate(track_ids, high, detections, now)

        # Stage 2: still-unmatched tracks vs low-confidence detections (continuation only)
        remaining_tracks = [tid for tid in track_ids if tid not in m_tracks]
        m2_tracks, m2_dets = self._associate(remaining_tracks, low, detections, now)
        m_tracks |= m2_tracks
        m_dets |= m2_dets

        # Stage 3: confirmed tracks that still have nothing vs leftover high-confidence detections,
        # by distance (moving people)
        remaining_tracks = [tid for tid in track_ids if tid not in m_tracks and self.tracks[tid].status == "confirmed"]
        remaining_high = [di for di in high if di not in m_dets]
        m3_tracks, m3_dets = self._associate_by_distance(remaining_tracks, remaining_high, detections, now)
        m_tracks |= m3_tracks
        m_dets |= m3_dets

        # Missed tracks
        for tid in track_ids:
            if tid not in m_tracks:
                self.tracks[tid].mark_missed(now=now, lost_seconds=self.lost_seconds)

        # New tracks only from unmatched HIGH-confidence detections
        for di in high:
            if di in m_dets:
                continue
            det = detections[di]
            self.tracks[self.next_track_id] = Track(self.next_track_id, det.bbox, det.confidence, now=now)
            self.next_track_id += 1

        self._merge_duplicate_tracks()

        # Drop lost tracks; remember the confirmed ones so the counter can emit EXIT events
        for tid in [tid for tid, trk in self.tracks.items() if trk.status == "lost"]:
            trk = self.tracks.pop(tid)
            if trk.confirmed_at is not None:
                self.exited_tracks.append(trk)

        return self.get_counted_tracks()

    def get_counted_tracks(self) -> List[Track]:
        return [trk for trk in self.tracks.values() if trk.status == "confirmed"]

    def _merge_duplicate_tracks(self):
        """
        If a person's box changes shape a new track can be spawned on top of the old one, leaving two
        tracks (and two ENTRY events) for one person.  Only genuine duplicates are removed:

          * a tentative track overlapping any older track (it never earned a place), or
          * a confirmed track that is currently NOT detected whose box has been taken over by a
            live track of about the same size and place (IoU, not containment).

        A confirmed track that is being detected right now is never dropped: a seated person whose
        box lies inside a standing person's box is an occlusion, not a duplicate.
        """
        ordered = sorted(self.tracks.values(),
                         key=lambda t: (t.status == "confirmed", t.frames_seen, -t.track_id), reverse=True)
        drop = set()
        for i, a in enumerate(ordered):
            if a.track_id in drop:
                continue
            for b in ordered[i + 1:]:
                if b.track_id in drop:
                    continue
                # Both matched to separate detections this frame -> two people (the detector has
                # already removed nested duplicate boxes), never a duplicate track.
                if a.consecutive_misses == 0 and b.consecutive_misses == 0:
                    continue
                if b.status == "tentative":
                    if match_score(a.bbox, b.bbox) > self.duplicate_overlap:
                        drop.add(b.track_id)
                elif b.consecutive_misses > 0 and a.consecutive_misses == 0:
                    if iou(a.bbox, b.bbox) > self.duplicate_overlap:
                        drop.add(b.track_id)
        for tid in drop:
            trk = self.tracks.pop(tid)
            # A confirmed duplicate already produced an ENTRY; balance it with an EXIT
            if trk.confirmed_at is not None:
                trk.lost_reason = "merged"
                self.exited_tracks.append(trk)


# ============================================================================
# Analytics
# ============================================================================

class CountingEngine:
    """
    Per-camera occupancy accumulator and event engine (whole frame, no zones).

      * occupancy      = number of confirmed, not-lost tracks right now
      * ENTRY          = a track became confirmed (a person has been reliably present for ~1-2 s)
      * EXIT           = a confirmed track was lost (not seen for `lost_seconds`)
      * entries - exits == occupancy at all times
      * "Today's" totals reset at local midnight; they can be restored from the events table after
        a restart via `restore(entries, exits, peak)`.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id

        self.current_occupancy: int = 0
        self.total_entries: int = 0
        self.total_exits: int = 0
        self.peak_occupancy: int = 0
        self.day_key: str = self._today()

        self.known_track_ids = set()   # confirmed tracks that have been counted as ENTRY

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def restore(self, entries: int, exits: int, peak: int):
        """Seed today's totals (e.g. from the database after a restart)."""
        self.total_entries = int(entries)
        self.total_exits = int(exits)
        self.peak_occupancy = int(peak)
        logger.info(f"[{self.camera_id}] Restored today's totals: entries={entries} exits={exits} peak={peak}")

    def _roll_day_if_needed(self):
        today = self._today()
        if today != self.day_key:
            logger.info(f"[{self.camera_id}] New day {today}: resetting daily totals "
                        f"(yesterday: entries={self.total_entries} exits={self.total_exits} peak={self.peak_occupancy})")
            self.day_key = today
            self.total_entries = 0
            self.total_exits = 0
            self.peak_occupancy = self.current_occupancy

    def _event(self, trk: Track, event_type: str, now: float) -> Dict[str, Any]:
        return {
            "event_id": f"{self.camera_id}_{trk.track_id}_{event_type}_{int(now)}",
            "camera_id": self.camera_id,
            "track_id": trk.track_id,
            "event_type": event_type,
            "timestamp": now,
            "confidence": trk.mean_confidence,
            "current_occupancy": self.current_occupancy,
        }

    def process_tracks(self, tracks: List[Track], exited: Optional[List[Track]] = None) -> List[Dict[str, Any]]:
        events = []
        now = time.time()
        self._roll_day_if_needed()

        self.current_occupancy = len(tracks)

        # Exits first so the occupancy recorded on each event is consistent
        for trk in (exited or []):
            if trk.track_id in self.known_track_ids:
                self.known_track_ids.discard(trk.track_id)
                self.total_exits += 1
                events.append(self._event(trk, "EXIT", now))
                logger.info(f"[{self.camera_id}] Track {trk.track_id} -> EXIT (occupancy: {self.current_occupancy}, "
                            f"reason {trk.lost_reason}, lived {now - (trk.confirmed_at or now):.0f}s)")

        for trk in tracks:
            trk.zone = "workspace"
            if trk.track_id not in self.known_track_ids:
                self.known_track_ids.add(trk.track_id)
                self.total_entries += 1
                events.append(self._event(trk, "ENTRY", now))
                logger.info(f"[{self.camera_id}] Track {trk.track_id} -> ENTRY "
                            f"(occupancy: {self.current_occupancy}, mean conf {trk.mean_confidence:.2f}, "
                            f"hit ratio {trk.hit_ratio:.2f})")

        if self.current_occupancy > self.peak_occupancy:
            self.peak_occupancy = self.current_occupancy

        return events

    def get_stats(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "current_occupancy": self.current_occupancy,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "peak_occupancy": self.peak_occupancy,
            "day": self.day_key,
        }


class ExclusionZones:
    """
    Per-camera "don't count anything here" regions.

    Each zone is a polygon in the camera's own pixel coordinates (the native stream resolution, not
    the size the dashboard happens to display).  A detection is discarded when the centre of its box
    falls inside any zone, which is applied before tracking, so an excluded region produces no
    tracks, no ENTRY/EXIT events and nothing on the debug overlay.

    Typical use: a doorway showing the next room, a reflective glass wall, a corridor that belongs to
    a different camera's count.
    """

    def __init__(self, zones: Optional[Sequence[Any]] = None):
        self.polygons: List[np.ndarray] = []
        self.set_zones(zones)

    # ------------------------------------------------------------------ config
    @staticmethod
    def normalise(zones: Optional[Sequence[Any]]) -> List[List[List[int]]]:
        """Accept a single polygon or a list of polygons; drop anything with fewer than 3 points."""
        if not zones:
            return []
        # A single polygon was given as [[x, y], [x, y], ...]
        if isinstance(zones[0], (list, tuple)) and len(zones[0]) == 2 and isinstance(zones[0][0], (int, float)):
            zones = [zones]
        out = []
        for poly in zones:
            pts = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in poly if len(p) >= 2]
            if len(pts) >= 3:
                out.append(pts)
        return out

    def set_zones(self, zones: Optional[Sequence[Any]]) -> None:
        self.polygons = [np.array(p, dtype=np.int32) for p in self.normalise(zones)]

    @property
    def enabled(self) -> bool:
        return bool(self.polygons)

    # ------------------------------------------------------------------ use
    def contains_point(self, x: float, y: float) -> bool:
        pt = (float(x), float(y))
        return any(cv2.pointPolygonTest(poly, pt, False) >= 0 for poly in self.polygons)

    def excludes_bbox(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox
        return self.contains_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def filter_detections(self, detections: List) -> List:
        if not self.polygons:
            return detections
        return [d for d in detections if not self.excludes_bbox(d.bbox)]

    def draw(self, frame: np.ndarray, colour=(80, 80, 200), alpha: float = 0.25) -> None:
        """Shade the excluded regions on the debug overlay so it is obvious what is being ignored."""
        if not self.polygons:
            return
        overlay = frame.copy()
        cv2.fillPoly(overlay, self.polygons, colour)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, self.polygons, True, colour, 2)
        for poly in self.polygons:
            x, y = poly[0]
            cv2.putText(frame, "EXCLUDED", (int(x), max(14, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)


# ============================================================================
# Camera stream reading
# ============================================================================

class ThreadedStreamReader:
    """
    Threaded Camera / Video Stream Reader.
    Supports RTSP, HTTP, USB Webcams, MP4 files.
    Includes exponential backoff auto-reconnection and FPS tracking.
    """
    def __init__(self, camera_id: str, source: str):
        self.camera_id = camera_id
        self.source = int(source) if str(source).isdigit() else source

        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.is_connected: bool = False
        self.stopped: bool = False

        self.fps: float = 0.0
        self.frame_count: int = 0
        self.start_time: float = time.time()

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _connect(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.is_connected = True
                logger.info(f"[{self.camera_id}] Successfully connected to source: {self.source}")
                return True
        except Exception as e:
            logger.error(f"[{self.camera_id}] Failed to connect: {e}")

        self.is_connected = False
        return False

    def _reader_loop(self):
        retry_delay = 1.0
        max_delay = 30.0

        while not self.stopped:
            if not self.is_connected:
                if not self._connect():
                    time.sleep(retry_delay)
                    retry_delay = min(max_delay, retry_delay * 2)
                    continue
                else:
                    retry_delay = 1.0

            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed > 1.0:
                        self.fps = round(self.frame_count / elapsed, 1)
                        self.frame_count = 0
                        self.start_time = time.time()
            else:
                with self.lock:
                    self.is_connected = False
                logger.warning(f"[{self.camera_id}] Frame read failed. Retrying connection...")
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.is_connected or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        logger.info(f"[{self.camera_id}] Stream reader released.")


# ============================================================================
# WebSocket broadcast (live dashboard updates)
# ============================================================================

class ConnectionManager:
    """
    WebSocket Connection Manager for live dashboard real-time updates.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        disconnected = []
        payload = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


ws_manager = ConnectionManager()


# ============================================================================
# Per-camera pipeline worker: read -> detect -> track -> count -> annotate
# ============================================================================

class CameraPipelineWorker:
    def __init__(self, camera_cfg: Dict[str, Any], event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 detector: Optional[YOLOPersonDetector] = None):
        self.camera_id = camera_cfg["camera_id"]
        self.camera_name = camera_cfg.get("name", self.camera_id)
        self.stream_url = camera_cfg["stream_url"]
        self.confidence_threshold = float(camera_cfg.get("confidence_threshold", settings.CONFIDENCE_THRESHOLD))
        self.event_loop = event_loop

        self.reader = ThreadedStreamReader(self.camera_id, self.stream_url)
        # Detector is shared across all cameras (one model in memory, serialised inference)
        self.detector = detector or YOLOPersonDetector(
            model_path=settings.MODEL_PATH,
            confidence_threshold=self.confidence_threshold,
            device=settings.MODEL_DEVICE,
            img_size=settings.IMAGE_SIZE
        )
        self.tracker = MultiObjectTracker(
            new_track_confidence=self.confidence_threshold,
            confirm_seconds=settings.TRACK_CONFIRM_SECONDS,
            confirm_min_hits=settings.TRACK_CONFIRM_MIN_HITS,
            confirm_hit_ratio=settings.TRACK_CONFIRM_HIT_RATIO,
            confirm_mean_conf=settings.TRACK_CONFIRM_MEAN_CONF,
            confirm_peak_conf=settings.TRACK_CONFIRM_PEAK_CONF,
            lost_seconds=settings.TRACK_LOST_SECONDS,
            reference_mode=settings.REFERENCE_POINT
        )
        self.counting_engine = CountingEngine(camera_id=self.camera_id)
        self._restore_daily_totals()

        # Regions the operator has marked "do not count here" (camera pixel coordinates)
        self.exclusion = ExclusionZones(camera_cfg.get("exclude_zones"))

        # Native stream size, published so the dashboard editor can map clicks back to camera pixels
        self.frame_width: int = 0
        self.frame_height: int = 0

        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.frame_seq: int = 0            # increments per processed frame; video streams wait on it
        self.last_processed_at: float = 0.0
        self._fps_window_start: float = time.time()
        self._fps_window_frames: int = 0
        self.processing_fps: float = 0.0
        self.detection_latency_ms: float = 0.0
        self.tracking_latency_ms: float = 0.0
        self.end_to_end_latency_ms: float = 0.0

        self.stopped = False
        self.thread: Optional[threading.Thread] = None

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        return self.latest_annotated_frame

    def set_exclude_zones(self, zones):
        """Apply new exclusion regions live - no restart, no reset of today's counters."""
        self.exclusion.set_zones(zones)
        logger.info(f"[{self.camera_id}] exclusion zones updated: {len(self.exclusion.polygons)} region(s)")

    def _restore_daily_totals(self):
        """After a restart, today's entries/exits/peak come back from the events table."""
        try:
            db = SessionLocal()
            try:
                # A previous run may have been killed without closing its tracks; balance the log first
                orphans = EventRepository.close_orphan_sessions(db, self.camera_id)
                if orphans:
                    logger.warning(f"[{self.camera_id}] closed {orphans} session(s) left open by a previous run")
                totals = EventRepository.get_daily_totals(db, self.camera_id)
            finally:
                db.close()
            if not totals or not (totals["entries"] or totals["exits"]):
                return
            net = totals["entries"] - totals["exits"]
            if net > settings.MAX_PLAUSIBLE_OCCUPANCY or net < 0:
                # entries - exits must equal the number of people present. A history that breaks that
                # (e.g. events written by the old counter, which never emitted EXITs) is not trusted.
                logger.warning(f"[{self.camera_id}] today's stored events are inconsistent "
                               f"(entries={totals['entries']} exits={totals['exits']}); starting today's totals from 0")
                return
            self.counting_engine.restore(totals["entries"], totals["exits"], totals["peak"])
        except Exception as e:
            logger.warning(f"[{self.camera_id}] could not restore daily totals: {e}")

    def start(self):
        """Run this camera's pipeline in its own thread."""
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, name=f"worker-{self.camera_id}", daemon=True)
            self.thread.start()

    def _loop(self):
        min_interval = 1.0 / max(1, settings.TARGET_PROCESSING_FPS)
        while not self.stopped:
            t = time.time()
            try:
                self.run_step()
            except Exception as e:
                logger.exception(f"[{self.camera_id}] pipeline step failed: {e}")
                time.sleep(0.5)
            # Cap per-camera rate; with many cameras the shared detector lock is the real limiter
            time.sleep(max(0.005, min_interval - (time.time() - t)))

    def run_step(self):
        ok, frame = self.reader.read()
        if not ok or frame is None:
            return

        t_start = time.time()

        # 1. Person Detection
        t0 = time.time()
        # Detector floor is low; the tracker decides what is confident enough to become a person
        detections = self.detector.detect(frame, confidence_threshold=min(settings.DETECTION_FLOOR, self.confidence_threshold))
        t1 = time.time()
        self.detection_latency_ms = (t1 - t0) * 1000.0

        # 1b. Drop anything inside an excluded region before it can become a track
        detections = self.exclusion.filter_detections(detections)

        # 2. Multi-Object Tracking
        t2 = time.time()
        confirmed_tracks = self.tracker.update(detections)
        t3 = time.time()
        self.tracking_latency_ms = (t3 - t2) * 1000.0

        # 3. Spatial & Counting Analysis
        events = self.counting_engine.process_tracks(confirmed_tracks, exited=self.tracker.exited_tracks)

        # 4. Save events to DB idempotently & broadcast WS update
        if events:
            db = SessionLocal()
            try:
                for evt in events:
                    EventRepository.insert_event_idempotent(db, evt)
            finally:
                db.close()

            if self.event_loop and self.event_loop.is_running():
                stats = self.counting_engine.get_stats()
                stats["type"] = "OCCUPANCY_UPDATE"
                stats["camera_id"] = self.camera_id
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast(stats), self.event_loop)

        # 5. Visual Debug Overlay Annotation
        h, w = frame.shape[:2]
        self.frame_height, self.frame_width = h, w
        self.exclusion.draw(frame)

        # Draw Active Tracks.
        # The label is the CURRENT detection score, not the track's lifetime average: a box can be
        # held open by a weak detection (a seated person who is barely visible), and labelling it
        # with the average made it look like the model was confident about something it was not.
        # Amber marks a track that is coasting - no detection matched it in this frame.
        for trk in confirmed_tracks:
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]
            coasting = trk.consecutive_misses > 0
            color = (0, 191, 255) if coasting else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            ref_x, ref_y = [int(v) for v in trk.current_point]
            cv2.circle(frame, (ref_x, ref_y), 4, (0, 0, 255), -1)

            label = (f"ID #{trk.track_id} (no detection)" if coasting
                     else f"ID #{trk.track_id} ({trk.confidence:.2f})")
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw KPI Header Overlay
        stats = self.counting_engine.get_stats()
        cv2.putText(frame, f"Occupancy: {stats['current_occupancy']}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        t_end = time.time()
        self.end_to_end_latency_ms = (t_end - t_start) * 1000.0
        # Frames actually completed per wall-clock second (includes any wait between frames), which is
        # the rate that matters. Measuring 1/(work time) instead would ignore the gaps and overstate it.
        self._fps_window_frames += 1
        window = t_end - self._fps_window_start
        if window >= 2.0:
            self.processing_fps = round(self._fps_window_frames / window, 1)
            self._fps_window_start = t_end
            self._fps_window_frames = 0
        self.latest_annotated_frame = frame
        self.frame_seq += 1
        self.last_processed_at = t_end

    def _flush_open_tracks(self):
        """
        On shutdown, close out everyone still being tracked with an EXIT event.

        Without this, a confirmed track that is alive when the process stops never gets its EXIT, so
        the stored event log says those people are still in the room. Replaying the day's events then
        drifts upwards with every restart, which corrupts the daily average and peak.
        """
        try:
            open_tracks = self.tracker.get_counted_tracks()
            if not open_tracks:
                return
            for trk in open_tracks:
                trk.lost_reason = "shutdown"
            events = self.counting_engine.process_tracks([], exited=open_tracks)
            if events:
                db = SessionLocal()
                try:
                    for evt in events:
                        EventRepository.insert_event_idempotent(db, evt)
                finally:
                    db.close()
            logger.info(f"[{self.camera_id}] closed {len(open_tracks)} open track(s) on shutdown")
        except Exception as e:
            logger.warning(f"[{self.camera_id}] could not close open tracks: {e}")

    def stop(self):
        self.stopped = True
        self._flush_open_tracks()
        self.reader.release()
        if self.thread is not None and self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=3.0)


# ============================================================================
# Camera manager: watches cameras.json, owns the per-camera workers
# ============================================================================

class CameraManager:
    """
    Owns one CameraPipelineWorker (and its thread) per enabled camera in cameras.json and keeps
    them in sync with the file: cameras added, removed, disabled, or whose stream URL / threshold
    changed are started, stopped or rebuilt live - no restart needed. All workers share one small
    pool of detector instances, so memory stays bounded however many cameras are added.
    """
    # Changing any of these requires rebuilding the worker
    REBUILD_KEYS = ("stream_url", "confidence_threshold")

    def __init__(self, workers: Dict[str, CameraPipelineWorker], event_loop=None, poll_seconds: float = 2.0):
        self.workers = workers                  # shared dict the API routes read from
        self.event_loop = event_loop
        self.poll_seconds = poll_seconds
        self._cfg_snapshot: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stopped = False

        logger.info(f"Loading shared detector: {settings.MODEL_PATH} @ {settings.IMAGE_SIZE}px on "
                    f"{settings.MODEL_DEVICE}, {settings.DETECTOR_POOL_SIZE} concurrent inference(s)")
        self.detector = build_detector()

    # ------------------------------------------------------------------ config
    def _load_config(self) -> List[Dict[str, Any]]:
        if not os.path.exists(settings.CAMERAS_CONFIG_PATH):
            return []
        try:
            with open(settings.CAMERAS_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            return cfg if isinstance(cfg, list) else []
        except Exception as e:
            # A half-written file (dashboard save in progress) should not tear workers down
            logger.error(f"Could not read {settings.CAMERAS_CONFIG_PATH}: {e}")
            return list(self._cfg_snapshot.values())

    # ------------------------------------------------------------------ lifecycle
    def _start_worker(self, cfg: Dict[str, Any], previous: CameraPipelineWorker = None):
        cid = cfg["camera_id"]
        logger.info(f"Starting pipeline worker for camera {cid} ({cfg.get('name', '')}) -> {cfg.get('stream_url')}")
        worker = CameraPipelineWorker(cfg, event_loop=self.event_loop, detector=self.detector)
        if previous is not None:
            # Editing a camera (new URL / threshold) must not reset today's counters or re-count
            # the people already in view: hand the tracker and counting engine over.
            worker.tracker = previous.tracker
            worker.counting_engine = previous.counting_engine
        worker.start()
        self.workers[cid] = worker
        self._cfg_snapshot[cid] = dict(cfg)

    def _stop_worker(self, cid: str, reason: str) -> CameraPipelineWorker:
        worker = self.workers.pop(cid, None)
        self._cfg_snapshot.pop(cid, None)
        if worker is not None:
            logger.info(f"Stopping pipeline worker for camera {cid} ({reason})")
            try:
                worker.stop()
            except Exception as e:
                logger.warning(f"[{cid}] error while stopping: {e}")
        return worker

    def sync(self):
        """Reconcile running workers with the config file."""
        cams = self._load_config()
        wanted = {
            c["camera_id"]: c for c in cams
            if c.get("camera_id") and c.get("enabled", True) and c.get("stream_url")
        }

        with self._lock:
            for cid in list(self.workers.keys()):
                if cid not in wanted:
                    self._stop_worker(cid, "removed or disabled in config")

            for cid, cfg in wanted.items():
                if cid not in self.workers:
                    self._start_worker(cfg)
                    continue
                old = self._cfg_snapshot.get(cid, {})
                if any(cfg.get(k) != old.get(k) for k in self.REBUILD_KEYS):
                    previous = self._stop_worker(cid, "stream settings changed")
                    self._start_worker(cfg, previous=previous)
                else:
                    # Settings that can be applied to a running worker, keeping today's counters
                    if cfg.get("exclude_zones") != old.get("exclude_zones"):
                        self.workers[cid].set_exclude_zones(cfg.get("exclude_zones"))
                    if cfg.get("name") != old.get("name"):
                        self.workers[cid].camera_name = cfg.get("name", cid)
                    self._cfg_snapshot[cid] = dict(cfg)

    def run_forever(self):
        logger.info("Camera manager started; watching cameras.json for changes")
        while not self._stopped:
            try:
                self.sync()
                workers = list(self.workers.values())
                occupancy = sum(worker.counting_engine.current_occupancy for worker in workers)
                cameras_ready = bool(workers) and all(worker.reader.is_connected for worker in workers)
                light_automation.update(occupancy, cameras_ready)
            except Exception as e:
                logger.exception(f"Camera manager sync failed: {e}")
            time.sleep(self.poll_seconds)

    def start(self):
        threading.Thread(target=self.run_forever, name="camera-manager", daemon=True).start()

    def stop(self):
        self._stopped = True
        with self._lock:
            for cid in list(self.workers.keys()):
                self._stop_worker(cid, "shutdown")


# ============================================================================
# API routes - cameras (CRUD + exclusion zones)
# ============================================================================

cameras_router = APIRouter(prefix="/cameras", tags=["Cameras"])


def get_cameras_file_config() -> List[Dict[str, Any]]:
    if os.path.exists(settings.CAMERAS_CONFIG_PATH):
        with open(settings.CAMERAS_CONFIG_PATH, "r") as f:
            return json.load(f)
    return []


def save_cameras_file_config(cfg: List[Dict[str, Any]]):
    with open(settings.CAMERAS_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _find_camera(cameras: List[Dict[str, Any]], camera_id: str) -> int:
    for i, c in enumerate(cameras):
        if c["camera_id"] == camera_id:
            return i
    raise HTTPException(status_code=404, detail="Camera not found")


@cameras_router.get("/{camera_id}/exclude-zones")
def get_exclude_zones(camera_id: str):
    cameras = get_cameras_file_config()
    cam = cameras[_find_camera(cameras, camera_id)]
    return {"camera_id": camera_id, "zones": ExclusionZones.normalise(cam.get("exclude_zones"))}


@cameras_router.put("/{camera_id}/exclude-zones")
def set_exclude_zones(camera_id: str, payload: Dict[str, Any] = Body(...)):
    """
    Replace the camera's exclusion regions.

    Body: {"zones": [[[x, y], [x, y], [x, y], ...], ...]} in the camera's native pixel coordinates.
    Polygons with fewer than 3 points are dropped; an empty list clears all regions.
    The running detector picks the change up within a couple of seconds - no restart.
    """
    zones = ExclusionZones.normalise(payload.get("zones"))
    cameras = get_cameras_file_config()
    idx = _find_camera(cameras, camera_id)
    if zones:
        cameras[idx]["exclude_zones"] = zones
    else:
        cameras[idx].pop("exclude_zones", None)
    save_cameras_file_config(cameras)
    return {"camera_id": camera_id, "zones": zones, "message": f"{len(zones)} exclusion region(s) saved"}


@cameras_router.get("", response_model=List[Dict[str, Any]])
def list_cameras():
    return get_cameras_file_config()


@cameras_router.get("/{camera_id}")
def get_camera(camera_id: str):
    cameras = get_cameras_file_config()
    for c in cameras:
        if c["camera_id"] == camera_id:
            return c
    raise HTTPException(status_code=404, detail="Camera not found")


@cameras_router.post("", status_code=201)
def create_camera(camera_data: Dict[str, Any] = Body(...)):
    cameras = get_cameras_file_config()
    cid = camera_data.get("camera_id")
    if not cid:
        raise HTTPException(status_code=400, detail="camera_id required")

    for c in cameras:
        if c["camera_id"] == cid:
            raise HTTPException(status_code=400, detail="Camera ID already exists")

    cameras.append(camera_data)
    save_cameras_file_config(cameras)
    return {"message": "Camera created", "camera": camera_data}


@cameras_router.put("/{camera_id}")
def update_camera(camera_id: str, camera_data: Dict[str, Any] = Body(...)):
    cameras = get_cameras_file_config()
    found = False
    for i, c in enumerate(cameras):
        if c["camera_id"] == camera_id:
            cameras[i].update(camera_data)
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Camera not found")
    save_cameras_file_config(cameras)
    return {"message": "Camera updated", "camera_id": camera_id}


@cameras_router.delete("/{camera_id}")
def delete_camera(camera_id: str):
    cameras = get_cameras_file_config()
    new_cams = [c for c in cameras if c["camera_id"] != camera_id]
    if len(new_cams) == len(cameras):
        raise HTTPException(status_code=404, detail="Camera not found")
    save_cameras_file_config(new_cams)
    return {"message": "Camera deleted", "camera_id": camera_id}


# ============================================================================
# API routes - occupancy (building-wide + per camera stats)
# ============================================================================

occupancy_router = APIRouter(prefix="/occupancy", tags=["Occupancy"])
lighting_router = APIRouter(prefix="/lighting", tags=["Lighting"])

# In-memory worker statistics registry reference set by main pipeline
_occupancy_workers: Dict[str, CameraPipelineWorker] = {}

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


def register_occupancy_workers(workers_dict):
    global _occupancy_workers
    _occupancy_workers = workers_dict


def _camera_stats(cid: str, worker: CameraPipelineWorker) -> Dict[str, Any]:
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

    for cid, worker in list(_occupancy_workers.items()):
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


@occupancy_router.get("")
def get_total_occupancy():
    return get_summary()


@occupancy_router.get("/{camera_id}")
def get_camera_occupancy(camera_id: str):
    if camera_id in _occupancy_workers:
        return _camera_stats(camera_id, _occupancy_workers[camera_id])
    return {"camera_id": camera_id, "current_occupancy": 0, "total_entries": 0, "total_exits": 0, "peak_occupancy": 0}


@lighting_router.get("")
def get_lighting_status():
    workers = list(_occupancy_workers.values())
    status = light_automation.get_status()
    status.update({
        "current_occupancy": sum(worker.counting_engine.current_occupancy for worker in workers),
        "cameras_online": sum(1 for worker in workers if worker.reader.is_connected),
        "cameras_total": len(workers),
    })
    return status


@lighting_router.put("")
def configure_lighting(payload: Dict[str, Any] = Body(...)):
    allowed = {"enabled", "mode", "off_delay_seconds"}
    if set(payload) - allowed:
        raise HTTPException(status_code=400, detail="Only enabled, mode, and off_delay_seconds can be changed")
    if "enabled" in payload and not isinstance(payload["enabled"], bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")
    if "off_delay_seconds" in payload and (
        isinstance(payload["off_delay_seconds"], bool) or not isinstance(payload["off_delay_seconds"], int)
    ):
        raise HTTPException(status_code=400, detail="off_delay_seconds must be an integer")
    try:
        light_automation.configure(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return get_lighting_status()


# ============================================================================
# API routes - health & metrics
# ============================================================================

health_router = APIRouter(tags=["Health & Metrics"])

_health_workers: Dict[str, CameraPipelineWorker] = {}


def register_health_workers(workers_dict):
    global _health_workers
    _health_workers = workers_dict


@health_router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    cameras_status = {}
    for cid, worker in _health_workers.items():
        cameras_status[cid] = "ONLINE" if worker.reader.is_connected else "OFFLINE"

    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database": "connected" if db_ok else "disconnected",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cameras": cameras_status,
        "timestamp": time.time()
    }


@health_router.get("/metrics")
def get_metrics():
    metrics = []
    for cid, worker in _health_workers.items():
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


# ============================================================================
# API routes - video (MJPEG stream + snapshot)
# ============================================================================

video_router = APIRouter(prefix="/video", tags=["Video Feeds"])

_video_workers: Dict[str, CameraPipelineWorker] = {}

# Dashboard stream size / quality. With many cameras, JPEG encoding is a real CPU cost, so frames are
# downscaled, encoded once per processed frame, and the same bytes are served to every viewer.
STREAM_MAX_WIDTH = 960
STREAM_JPEG_QUALITY = 60

_encode_lock = threading.Lock()
_encoded_cache: Dict[str, Tuple[int, bytes]] = {}   # camera_id -> (frame_seq, jpeg bytes)


def register_video_workers(workers_dict):
    global _video_workers
    _video_workers = workers_dict


def _encoded_frame(camera_id: str) -> Optional[Tuple[int, bytes]]:
    worker = _video_workers.get(camera_id)
    if worker is None:
        return None
    frame = worker.get_annotated_frame()
    if frame is None:
        return None
    seq = worker.frame_seq

    cached = _encoded_cache.get(camera_id)
    if cached is not None and cached[0] == seq:
        return cached

    with _encode_lock:
        cached = _encoded_cache.get(camera_id)
        if cached is not None and cached[0] == seq:
            return cached
        h, w = frame.shape[:2]
        if w > STREAM_MAX_WIDTH:
            scale = STREAM_MAX_WIDTH / float(w)
            frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
        if not ok:
            return None
        result = (seq, encoded.tobytes())
        _encoded_cache[camera_id] = result
        return result


def generate_mjpeg_stream(camera_id: str):
    last_seq = -1
    while True:
        if camera_id not in _video_workers:
            time.sleep(0.5)
            continue
        item = _encoded_frame(camera_id)
        if item is not None and item[0] != last_seq:
            last_seq = item[0]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + item[1] + b'\r\n')
        time.sleep(0.05)


@video_router.get("/{camera_id}/stream")
def video_stream(camera_id: str):
    if camera_id not in _video_workers:
        raise HTTPException(status_code=404, detail="Camera stream worker not found")
    return StreamingResponse(generate_mjpeg_stream(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


@video_router.get("/{camera_id}/snapshot")
def video_snapshot(camera_id: str):
    """Single JPEG of the latest annotated frame - cheap for thumbnails / many tiles."""
    if camera_id not in _video_workers:
        raise HTTPException(status_code=404, detail="Camera stream worker not found")
    item = _encoded_frame(camera_id)
    if item is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=item[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ============================================================================
# FastAPI application
# ============================================================================

def create_app() -> FastAPI:
    fastapi_app = FastAPI(
        title=f"{settings.PROJECT_NAME} API",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_db()

    for router in (cameras_router, occupancy_router, lighting_router, health_router, video_router):
        fastapi_app.include_router(router, prefix=settings.API_PREFIX)

    @fastapi_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()   # keeps the connection open; we only broadcast
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    if os.path.exists(dashboard_dir):
        fastapi_app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    return fastapi_app


app = create_app()


# ============================================================================
# Entry point
# ============================================================================

# Shared registry: the API routes read live worker state from here
pipeline_workers: Dict[str, CameraPipelineWorker] = {}


def main():
    register_occupancy_workers(pipeline_workers)
    register_health_workers(pipeline_workers)
    register_video_workers(pipeline_workers)

    # One thread per camera, one shared detector, hot-reload of cameras.json
    manager = CameraManager(pipeline_workers)
    manager.start()

    logger.info("Starting FastAPI Uvicorn Server at http://0.0.0.0:8000  (dashboard: /dashboard)")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
