import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from database.connection import init_db
from api.websocket import ws_manager
from api.routes.cameras import router as cameras_router
from api.routes.occupancy import router as occupancy_router
from api.routes.health import router as health_router
from api.routes.video import router as video_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=f"{settings.PROJECT_NAME} API",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_db()

    for router in (cameras_router, occupancy_router, health_router, video_router):
        app.include_router(router, prefix=settings.API_PREFIX)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()   # keeps the connection open; we only broadcast
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
    if os.path.exists(dashboard_dir):
        app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    return app

app = create_app()
