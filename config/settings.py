import os

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

settings = Settings()

os.makedirs("config", exist_ok=True)
