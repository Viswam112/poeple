import os
import json
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Body
from config.settings import settings
from core.analytics.exclusion import ExclusionZones

router = APIRouter(prefix="/cameras", tags=["Cameras"])

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


@router.get("/{camera_id}/exclude-zones")
def get_exclude_zones(camera_id: str):
    cameras = get_cameras_file_config()
    cam = cameras[_find_camera(cameras, camera_id)]
    return {"camera_id": camera_id, "zones": ExclusionZones.normalise(cam.get("exclude_zones"))}


@router.put("/{camera_id}/exclude-zones")
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


@router.get("", response_model=List[Dict[str, Any]])
def list_cameras():
    return get_cameras_file_config()

@router.get("/{camera_id}")
def get_camera(camera_id: str):
    cameras = get_cameras_file_config()
    for c in cameras:
        if c["camera_id"] == camera_id:
            return c
    raise HTTPException(status_code=404, detail="Camera not found")

@router.post("", status_code=201)
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

@router.put("/{camera_id}")
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

@router.delete("/{camera_id}")
def delete_camera(camera_id: str):
    cameras = get_cameras_file_config()
    new_cams = [c for c in cameras if c["camera_id"] != camera_id]
    if len(new_cams) == len(cameras):
        raise HTTPException(status_code=404, detail="Camera not found")
    save_cameras_file_config(new_cams)
    return {"message": "Camera deleted", "camera_id": camera_id}
