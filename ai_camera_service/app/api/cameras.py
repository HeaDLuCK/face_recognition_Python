from html import escape
import asyncio
import time
from urllib.parse import quote

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class ChannelDiscoveryRequest(BaseModel):
    tenantId: str | None = None
    rtspUrl: str | None = None
    ip: str | None = None
    username: str | None = None
    password: str | None = None
    rtspPort: int = 554
    channels: list[str] | None = None
    maxCamera: int = Field(default=16, ge=1, le=64)
    timeoutSeconds: int = Field(default=4, ge=1, le=20)


class ChannelDiscoveryResult(BaseModel):
    channel: str
    rtspUrl: str
    width: int
    height: int


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


@router.post("/discover-channels")
async def discover_channels(payload: ChannelDiscoveryRequest) -> dict:
    base_url = _channel_base_url(payload)
    candidate_channels = payload.channels or _default_hikvision_channels(payload.maxCamera)
    results = []

    for channel in candidate_channels:
        rtsp_url = f"{base_url}/{channel}"
        opened, width, height = await asyncio.to_thread(
            _test_rtsp_channel,
            rtsp_url,
            payload.timeoutSeconds,
        )
        if opened:
            results.append(
                ChannelDiscoveryResult(
                    channel=channel,
                    rtspUrl=rtsp_url,
                    width=width,
                    height=height,
                ).model_dump()
            )

    return {
        "tenantId": payload.tenantId,
        "count": len(results),
        "workingChannels": results,
        "rtspChannels": [item["channel"] for item in results],
        "envValue": ",".join(item["channel"] for item in results),
    }


@router.get("/stream-flows")
async def list_stream_flows(request: Request) -> dict:
    return {
        "count": len(request.app.state.runtime_state.list_cameras()),
        "streams": [
            _stream_flow(request, camera.cameraId)
            for camera in request.app.state.runtime_state.list_cameras()
        ],
    }


@router.get("/{cameraId}/stream-flow")
async def camera_stream_flow(cameraId: str, request: Request) -> dict:
    if request.app.state.runtime_state.get_camera(cameraId) is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found in synced ERP config. Run /api/sync/cameras first.",
        )
    return _stream_flow(request, cameraId)


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


@router.get("/{cameraId}/calibrate", response_class=HTMLResponse)
async def calibrate_camera(cameraId: str, request: Request) -> HTMLResponse:
    camera = request.app.state.runtime_state.get_camera(cameraId)
    if camera is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found in synced ERP config. Run /api/sync/cameras first.",
        )

    safe_camera_id = escape(cameraId)
    stream_camera_id = quote(cameraId, safe="")
    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Calibrate {safe_camera_id}</title>
            <style>
              :root {{
                color-scheme: dark;
                font-family: Arial, sans-serif;
                background: #111827;
                color: #f9fafb;
              }}
              body {{ margin: 0; background: #111827; }}
              header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 12px 16px;
                background: #0f172a;
                border-bottom: 1px solid #374151;
              }}
              h1 {{ margin: 0; font-size: 18px; }}
              .wrap {{ position: relative; display: inline-block; margin: 16px; }}
              img {{ display: block; max-width: calc(100vw - 32px); background: #000; }}
              canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }}
              .panel {{
                margin: 0 16px 16px;
                padding: 12px;
                border: 1px solid #374151;
                border-radius: 8px;
                background: #020617;
              }}
              code {{ color: #93c5fd; word-break: break-all; }}
              button {{
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 8px 10px;
                background: #111827;
                color: #f9fafb;
              }}
            </style>
          </head>
          <body>
            <header>
              <h1>{safe_camera_id} Calibration</h1>
              <div id="pos">x: -, y: -</div>
            </header>
            <div class="wrap">
              <img id="stream" src="/api/cameras/{stream_camera_id}/stream">
              <canvas id="canvas"></canvas>
            </div>
            <div class="panel">
              <button id="clear">Clear</button>
              <p>Click points around the watched area. Use the generated value in <code>MOTION_ZONES</code>.</p>
              <code id="zone">MOTION_ZONES=watch:</code>
            </div>
            <script>
              const img = document.getElementById("stream");
              const canvas = document.getElementById("canvas");
              const ctx = canvas.getContext("2d");
              const pos = document.getElementById("pos");
              const zone = document.getElementById("zone");
              const points = [];

              function resize() {{
                canvas.width = img.clientWidth;
                canvas.height = img.clientHeight;
                draw();
              }}

              function toImagePoint(event) {{
                const rect = canvas.getBoundingClientRect();
                const scaleX = img.naturalWidth / rect.width;
                const scaleY = img.naturalHeight / rect.height;
                return {{
                  x: Math.round((event.clientX - rect.left) * scaleX),
                  y: Math.round((event.clientY - rect.top) * scaleY),
                  sx: event.clientX - rect.left,
                  sy: event.clientY - rect.top
                }};
              }}

              function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (!points.length) return;
                ctx.strokeStyle = "#f59e0b";
                ctx.fillStyle = "rgba(245, 158, 11, 0.22)";
                ctx.lineWidth = 2;
                ctx.beginPath();
                points.forEach((point, index) => {{
                  const sx = point.x * canvas.width / img.naturalWidth;
                  const sy = point.y * canvas.height / img.naturalHeight;
                  if (index === 0) ctx.moveTo(sx, sy);
                  else ctx.lineTo(sx, sy);
                }});
                if (points.length > 2) ctx.closePath();
                ctx.stroke();
                if (points.length > 2) ctx.fill();
              }}

              function updateZone() {{
                zone.textContent = "MOTION_ZONES=watch:" + points.map(point => `${{point.x}},${{point.y}}`).join("|");
              }}

              canvas.addEventListener("mousemove", event => {{
                const point = toImagePoint(event);
                pos.textContent = `x: ${{point.x}}, y: ${{point.y}}`;
              }});

              canvas.addEventListener("click", event => {{
                const point = toImagePoint(event);
                points.push({{ x: point.x, y: point.y }});
                draw();
                updateZone();
              }});

              document.getElementById("clear").addEventListener("click", () => {{
                points.length = 0;
                draw();
                updateZone();
              }});

              img.addEventListener("load", resize);
              window.addEventListener("resize", resize);
            </script>
          </body>
        </html>
        """
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


def _channel_base_url(payload: ChannelDiscoveryRequest) -> str:
    if payload.rtspUrl:
        if "/Streaming/Channels/" in payload.rtspUrl:
            return payload.rtspUrl.rsplit("/", 1)[0]
        return payload.rtspUrl.rstrip("/")

    if not payload.ip or not payload.username or not payload.password:
        raise HTTPException(
            status_code=400,
            detail="Send either rtspUrl or ip + username + password.",
        )

    return f"rtsp://{payload.username}:{payload.password}@{payload.ip}:{payload.rtspPort}/Streaming/Channels"


def _default_hikvision_channels(max_camera: int) -> list[str]:
    channels = []
    for camera_number in range(1, max_camera + 1):
        channels.append(f"{camera_number}01")
        channels.append(f"{camera_number}02")
    return channels


def _test_rtsp_channel(rtsp_url: str, timeout_seconds: int) -> tuple[bool, int, int]:
    started_at = time.time()
    capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        while time.time() - started_at < timeout_seconds:
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                return True, int(width), int(height)
            time.sleep(0.1)
        return False, 0, 0
    finally:
        capture.release()


def _stream_flow(request: Request, camera_id: str) -> dict:
    return {
        "cameraId": camera_id,
        "type": "mjpeg",
        "method": "GET",
        "path": f"/api/cameras/{quote(camera_id, safe='')}/stream",
        "url": str(request.url_for("stream_camera", cameraId=camera_id)),
        "contentType": "multipart/x-mixed-replace; boundary=frame",
        "notes": "Use this application stream endpoint. Do not expose camera RTSP credentials to browser clients.",
    }
