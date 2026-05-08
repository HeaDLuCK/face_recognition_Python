from html import escape
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

router = APIRouter()


@router.get("/grid", response_class=HTMLResponse)
async def camera_grid(request: Request) -> HTMLResponse:
    cameras = request.app.state.runtime_state.list_cameras()
    status_by_id = {
        item["cameraId"]: item["status"]
        for item in request.app.state.camera_manager.status()["cameras"]
    }
    cards = "\n".join(
        _camera_card(
            camera_id=camera.cameraId,
            name=camera.name,
            tenant_id=camera.tenantId,
            status=status_by_id.get(camera.cameraId, "stopped"),
        )
        for camera in cameras
    )
    if not cards:
        cards = """
        <section class="empty">
          <h2>No synced cameras</h2>
          <p>Run POST /api/sync/cameras first, then refresh this page.</p>
        </section>
        """

    html = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AI Camera Grid</title>
        <style>
          :root {{
            color-scheme: dark;
            font-family: Arial, sans-serif;
            background: #111827;
            color: #f9fafb;
          }}
          body {{
            margin: 0;
            min-height: 100vh;
            background: #111827;
          }}
          header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 16px 20px;
            border-bottom: 1px solid #374151;
            background: #0f172a;
          }}
          h1 {{
            margin: 0;
            font-size: 20px;
            font-weight: 700;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 14px;
            padding: 14px;
          }}
          .camera {{
            overflow: hidden;
            border: 1px solid #374151;
            border-radius: 8px;
            background: #020617;
          }}
          .camera img {{
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            background: #000;
          }}
          .meta {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 12px;
          }}
          .name {{
            min-width: 0;
            font-size: 14px;
            font-weight: 700;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }}
          .sub {{
            color: #9ca3af;
            font-size: 12px;
          }}
          .badge {{
            flex: 0 0 auto;
            border: 1px solid #4b5563;
            border-radius: 999px;
            padding: 4px 8px;
            color: #d1d5db;
            font-size: 12px;
          }}
          .empty {{
            margin: 24px;
            padding: 20px;
            border: 1px solid #374151;
            border-radius: 8px;
            background: #020617;
          }}
          .empty h2 {{
            margin: 0 0 8px;
            font-size: 18px;
          }}
          .empty p {{
            margin: 0;
            color: #9ca3af;
          }}
        </style>
      </head>
      <body>
        <header>
          <h1>Camera Grid</h1>
          <span>{len(cameras)} camera(s)</span>
        </header>
        <main class="grid">{cards}</main>
      </body>
    </html>
    """
    return HTMLResponse(html)


@router.get("/{cameraId}/stream")
async def stream_camera(cameraId: str, request: Request) -> StreamingResponse:
    if request.app.state.runtime_state.get_camera(cameraId) is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found in synced ERP config. Run /api/sync/cameras first.",
        )
    generator = request.app.state.camera_manager.mjpeg_stream(cameraId)
    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/{cameraId}/start")
async def start_camera(cameraId: str, request: Request) -> dict:
    try:
        return await request.app.state.camera_manager.start_camera(cameraId)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{cameraId}/stop")
async def stop_camera(cameraId: str, request: Request) -> dict:
    return await request.app.state.camera_manager.stop_camera(cameraId)


@router.post("/start-all")
async def start_all(request: Request) -> dict:
    return await request.app.state.camera_manager.start_all()


@router.post("/stop-all")
async def stop_all(request: Request) -> dict:
    return await request.app.state.camera_manager.stop_all()


def _camera_card(camera_id: str, name: str, tenant_id: str, status: str) -> str:
    safe_camera_id = escape(camera_id)
    stream_camera_id = quote(camera_id, safe="")
    safe_name = escape(name)
    safe_tenant_id = escape(tenant_id)
    safe_status = escape(status)
    return f"""
    <article class="camera">
      <img src="/api/cameras/{stream_camera_id}/stream" alt="{safe_name}">
      <div class="meta">
        <div>
          <div class="name">{safe_name}</div>
          <div class="sub">{safe_tenant_id} / {safe_camera_id}</div>
        </div>
        <span class="badge">{safe_status}</span>
      </div>
    </article>
    """
