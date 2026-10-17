import time
import threading
import cv2
from typing import Dict, Tuple, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response

router = APIRouter(prefix="/video", tags=["Video Feeds"])

pipeline_workers_ref = {}

# Dashboard stream size / quality. With many cameras, JPEG encoding is a real CPU cost, so frames are
# downscaled, encoded once per processed frame, and the same bytes are served to every viewer.
STREAM_MAX_WIDTH = 960
STREAM_JPEG_QUALITY = 60

_encode_lock = threading.Lock()
_encoded_cache: Dict[str, Tuple[int, bytes]] = {}   # camera_id -> (frame_seq, jpeg bytes)


def register_workers_video(workers_dict):
    global pipeline_workers_ref
    pipeline_workers_ref = workers_dict


def _encoded_frame(camera_id: str) -> Optional[Tuple[int, bytes]]:
    worker = pipeline_workers_ref.get(camera_id)
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
        if camera_id not in pipeline_workers_ref:
            time.sleep(0.5)
            continue
        item = _encoded_frame(camera_id)
        if item is not None and item[0] != last_seq:
            last_seq = item[0]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + item[1] + b'\r\n')
        time.sleep(0.05)


@router.get("/{camera_id}/stream")
def video_stream(camera_id: str):
    if camera_id not in pipeline_workers_ref:
        raise HTTPException(status_code=404, detail="Camera stream worker not found")
    return StreamingResponse(generate_mjpeg_stream(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/{camera_id}/snapshot")
def video_snapshot(camera_id: str):
    """Single JPEG of the latest annotated frame - cheap for thumbnails / many tiles."""
    if camera_id not in pipeline_workers_ref:
        raise HTTPException(status_code=404, detail="Camera stream worker not found")
    item = _encoded_frame(camera_id)
    if item is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=item[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
