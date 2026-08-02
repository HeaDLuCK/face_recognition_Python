from html import escape
import asyncio
<<<<<<< HEAD
from pathlib import Path
from urllib.parse import quote
=======
import logging
from pathlib import Path
import time
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
<<<<<<< HEAD
from pydantic import BaseModel, Field
=======
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
from starlette.background import BackgroundTask

from app.cameras.hikvision_history import (
    delete_file_safely,
    export_hikvision_history_clip,
)
<<<<<<< HEAD
=======
from app.services.url_utils import redact_url_credentials
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054

router = APIRouter()
logger = logging.getLogger(__name__)


<<<<<<< HEAD
=======
class ChannelDiscoveryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tenantId: str | None = Field(default=None, validation_alias=AliasChoices("tenantId", "etsAuth"))
    cameraDeviceId: str | int | None = None
    defaultDirection: str = "BIDIRECTIONAL"
    defaultCapabilities: str = "FACE_RECOGNITION"
    defaultEnabled: bool = True
    libellePrefix: str = "Camera"
    rtspUrl: str | None = None
    ip: str | None = None
    username: str | None = None
    password: str | None = None
    rtspPort: int = 554
    rtspPath: str | None = None
    channels: list[str] | None = None
    maxCamera: int = Field(default=16, ge=1, le=64)
    timeoutSeconds: int = Field(default=2, ge=1, le=20)
    discoveryConcurrency: int = Field(default=8, ge=1, le=32)
    validateRtsp: bool = True
    includeOffline: bool = False


class ChannelDiscoveryResult(BaseModel):
    cameraId: str
    libelle: str
    opc_camera_device: str | int | None = None
    channel: str
    streamType: str
    direction: str
    capabilities: str
    enabled: bool
    status: str
    reachable: bool | None = None
    rtspUrl: str
    width: int
    height: int


>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
class HistoryClipRequest(BaseModel):
    timestamp: str
    beforeSeconds: int = Field(default=10, ge=0, le=300)
    afterSeconds: int = Field(default=10, ge=1, le=300)
    channel: str | None = None


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
            tenant_id=", ".join(camera.tenantIds),
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


<<<<<<< HEAD
=======
@router.post("/discover-channels")
async def discover_channels(payload: ChannelDiscoveryRequest) -> dict:
    base_url = _channel_base_url(payload)
    candidate_channels = payload.channels
    discovery_source = "request"


    if candidate_channels is None:
        candidate_channels = await asyncio.to_thread(_discover_channels_from_isapi, payload)
        discovery_source = "isapi"

    if not candidate_channels:
        candidate_channels = _default_hikvision_channels(payload.maxCamera)
        discovery_source = "rtsp_probe"

    if payload.validateRtsp or discovery_source == "rtsp_probe":
        semaphore = asyncio.Semaphore(payload.discoveryConcurrency)
        results = [
            result
            for result in await asyncio.gather(
                *[_discover_channel(payload, base_url, channel, semaphore) for channel in candidate_channels]
            )
            if result is not None
        ]
    else:
        results = [
            _channel_result(payload, base_url, channel, 0, 0, "UNKNOWN", None)
            for channel in candidate_channels
        ]

    working_results = [item for item in results if item.get("reachable") is True or item.get("status") == "ONLINE"]
    offline_results = [item for item in results if item not in working_results]
    camera_results = results if payload.includeOffline else working_results
    cameras = _group_discovered_cameras(camera_results, payload)

    return {
        "etsAuth": payload.tenantId,
        "cameraDeviceId": payload.cameraDeviceId,
        "discoverySource": discovery_source,
        "rtspValidated": payload.validateRtsp or discovery_source == "rtsp_probe",
        "count": len(cameras),
        "cameras": cameras,
        "workingChannels": working_results,
        "offlineChannels": offline_results,
        "checkedChannels": results,
        "rtspChannels": [item["channel"] for item in working_results],
        "envValue": ",".join(item["channel"] for item in working_results),
    }


async def _discover_channel(
    payload: ChannelDiscoveryRequest,
    base_url: str,
    channel: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    rtsp_url = _channel_rtsp_url(base_url, channel)
    async with semaphore:
        opened, width, height = await asyncio.to_thread(
            _test_rtsp_channel,
            rtsp_url,
            payload.timeoutSeconds,
        )

    if not opened:
        return _channel_result(payload, base_url, channel, 0, 0, "OFFLINE", False)

    return _channel_result(payload, base_url, channel, width, height, "ONLINE", True)


def _channel_result(
    payload: ChannelDiscoveryRequest,
    base_url: str,
    channel: str,
    width: int,
    height: int,
    status: str,
    reachable: bool | None,
) -> dict:
    rtsp_url = _channel_rtsp_url(base_url, channel)
    return ChannelDiscoveryResult(
        cameraId=channel,
        libelle=_camera_libelle(payload.libellePrefix, channel),
        opc_camera_device=payload.cameraDeviceId,
        channel=channel,
        streamType=_stream_type(channel),
        direction=payload.defaultDirection,
        capabilities=payload.defaultCapabilities,
        enabled=payload.defaultEnabled,
        status=status,
        reachable=reachable,
        rtspUrl=rtsp_url,
        width=width,
        height=height,
    ).model_dump()


def _group_discovered_cameras(results: list[dict], payload: ChannelDiscoveryRequest) -> list[dict]:
    grouped: dict[str, dict] = {}
    for item in results:
        channel = item["channel"]
        camera_number = _camera_number_from_channel(channel)
        camera = grouped.setdefault(
            camera_number,
            {
                "cameraId": item["cameraId"],
                "libelle": _camera_libelle_without_stream(payload.libellePrefix, camera_number),
                # "opc_camera_device": item["opc_camera_device"],
                "main_channel": None,
                "sub_channel": None,
                "direction": item["direction"],
                "capabilities": item["capabilities"],
                "enabled": item["enabled"],
                "status": item["status"],
                "reachable": item["reachable"],
            },
        )

        if item["streamType"] == "MAIN":
            camera["main_channel"] = channel
            camera["cameraId"] = channel
        elif item["streamType"] == "SUB":
            camera["sub_channel"] = channel

        camera["status"] = _combine_channel_status(camera["status"], item["status"])
        camera["reachable"] = _combine_channel_reachable(camera["reachable"], item["reachable"])

    return list(grouped.values())


def _combine_channel_status(current_status: str, next_status: str) -> str:
    priority = {"ONLINE": 3, "UNKNOWN": 2, "OFFLINE": 1}
    return current_status if priority.get(current_status, 0) >= priority.get(next_status, 0) else next_status


def _combine_channel_reachable(current_reachable: bool | None, next_reachable: bool | None) -> bool | None:
    if current_reachable is True or next_reachable is True:
        return True
    if current_reachable is False or next_reachable is False:
        return False
    return None


>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
@router.get("/stream-flows")
async def list_stream_flows(request: Request, streamType: str | None = None) -> dict:
    stream_type_filter = _normalize_stream_type_filter(streamType)
    cameras = request.app.state.runtime_state.list_cameras()
    if stream_type_filter != "ALL":
        cameras = [
            camera
            for camera in cameras
            if _stream_type_from_camera(camera) == stream_type_filter
        ]

    return {
        "streamType": stream_type_filter,
        "count": len(cameras),
        "streams": [
            _stream_flow(request, camera.cameraId, camera)
            for camera in cameras
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
async def stream_camera(cameraId: str, request: Request, overlay: bool = False) -> StreamingResponse:
    if request.app.state.runtime_state.get_camera(cameraId) is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found in synced ERP config. Run /api/sync/cameras first.",
        )
    generator = request.app.state.camera_manager.mjpeg_stream(cameraId, overlay=overlay)
    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/{cameraId}/debug-stream")
async def debug_stream_camera(cameraId: str, request: Request) -> StreamingResponse:
    if request.app.state.runtime_state.get_camera(cameraId) is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found in synced ERP config. Run /api/sync/cameras first.",
        )
    generator = request.app.state.camera_manager.mjpeg_stream(cameraId, overlay=True)
    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/{cameraId}/history-clip")
async def camera_history_clip(cameraId: str, payload: HistoryClipRequest, request: Request) -> FileResponse:
    camera = request.app.state.runtime_state.get_camera(cameraId)
    if camera is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found in synced ERP config. Run /api/sync/cameras first.",
        )

    try:
        clip_path = await asyncio.to_thread(
            export_hikvision_history_clip,
            camera.rtspUrl,
            payload.timestamp,
            payload.beforeSeconds,
            payload.afterSeconds,
            payload.channel,
            request.app.state.camera_manager.settings.history_clip_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=Path(clip_path).name,
        background=BackgroundTask(delete_file_safely, Path(clip_path)),
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
              <p>For <code>PERSON_COUNTING</code>, the first two clicks are the counting-line endpoints.</p>
              <code id="countingLine">"countingLine": null</code>
            </div>
            <script>
              const img = document.getElementById("stream");
              const canvas = document.getElementById("canvas");
              const ctx = canvas.getContext("2d");
              const pos = document.getElementById("pos");
              const zone = document.getElementById("zone");
              const countingLine = document.getElementById("countingLine");
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
                if (points.length >= 2 && img.naturalWidth && img.naturalHeight) {{
                  const line = {{
                    x1: Number((points[0].x / img.naturalWidth).toFixed(4)),
                    y1: Number((points[0].y / img.naturalHeight).toFixed(4)),
                    x2: Number((points[1].x / img.naturalWidth).toFixed(4)),
                    y2: Number((points[1].y / img.naturalHeight).toFixed(4)),
                    inSide: "POSITIVE",
                    hysteresis: 0.015
                  }};
                  countingLine.textContent = '"countingLine": ' + JSON.stringify(line);
                }} else {{
                  countingLine.textContent = '"countingLine": null';
                }}
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


def _stream_type(channel: str) -> str:
    return "MAIN" if channel.endswith("01") else "SUB"


def _normalize_stream_type_filter(stream_type: str | None) -> str:
    if not stream_type:
        return "ALL"

    normalized = stream_type.strip().upper()
    if normalized in {"ALL", "MAIN", "SUB"}:
        return normalized

    raise HTTPException(
        status_code=400,
        detail="streamType must be MAIN, SUB, or ALL.",
    )


def _stream_type_from_camera(camera) -> str:
    rtsp_url = getattr(camera, "rtspUrl", "")
    if "/Streaming/Channels/" not in rtsp_url and "/Streaming/tracks/" not in rtsp_url:
        return "MAIN"

    channel = _channel_from_camera(camera)
    return _stream_type(channel)


def _channel_from_camera(camera) -> str:
    rtsp_url = getattr(camera, "rtspUrl", "")
    if "/Streaming/Channels/" in rtsp_url:
        channel = rtsp_url.rsplit("/", 1)[-1]
    elif "/Streaming/tracks/" in rtsp_url:
        channel = rtsp_url.rsplit("/", 1)[-1]
    else:
        channel = getattr(camera, "cameraId", "")

    return channel.split("?", 1)[0].strip()


<<<<<<< HEAD
=======
def _camera_number_from_channel(channel: str) -> str:
    if channel.endswith(("01", "02")) and len(channel) > 2:
        return channel[:-2]
    return channel


def _camera_libelle_without_stream(prefix: str, camera_number: str) -> str:
    return f"{prefix} {camera_number}"


def _camera_libelle(prefix: str, channel: str) -> str:
    camera_number = channel[:-2] or channel
    return f"{prefix} {camera_number} {_stream_type(channel)}"


def _test_rtsp_channel(rtsp_url: str, timeout_seconds: int) -> tuple[bool, int, int]:
    started_at = time.time()
    timeout_ms = max(timeout_seconds * 1000, 1000)
    capture = cv2.VideoCapture(
        rtsp_url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            timeout_ms,
        ],
    )
    try:
        if not capture.isOpened():
            logger.warning(
                "RTSP_SOURCE=channel_discovery phase=open_failed stream=%s elapsed=%.2fs timeout=%ss",
                redact_url_credentials(rtsp_url),
                time.time() - started_at,
                timeout_seconds,
            )
            return False, 0, 0

        while time.time() - started_at < timeout_seconds:
            read_started_at = time.time()
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                return True, int(width), int(height)
            read_elapsed = time.time() - read_started_at
            if read_elapsed >= max(timeout_seconds * 0.8, 0.8):
                logger.warning(
                    "RTSP_SOURCE=channel_discovery phase=read_timeout stream=%s elapsed=%.2fs timeout=%ss",
                    redact_url_credentials(rtsp_url),
                    read_elapsed,
                    timeout_seconds,
                )
            time.sleep(0.1)
        logger.warning(
            "RTSP_SOURCE=channel_discovery phase=no_frame stream=%s elapsed=%.2fs timeout=%ss",
            redact_url_credentials(rtsp_url),
            time.time() - started_at,
            timeout_seconds,
        )
        return False, 0, 0
    finally:
        capture.release()


>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
def _stream_flow(request: Request, camera_id: str, camera=None) -> dict:
    if camera is None:
        camera = request.app.state.runtime_state.get_camera(camera_id)

    path = f"/api/cameras/{quote(camera_id, safe='')}/stream"
    debug_path = f"/api/cameras/{quote(camera_id, safe='')}/debug-stream"
    clean_url = str(request.url_for("stream_camera", cameraId=camera_id))
    return {
        "cameraId": camera_id,
        "streamType": _stream_type_from_camera(camera) if camera is not None else _stream_type(camera_id),
        "type": "mjpeg",
        "method": "GET",
        "path": path,
        "url": clean_url,
        "overlayPath": f"{path}?overlay=true",
        "overlayUrl": f"{clean_url}?overlay=true",
        "debugPath": debug_path,
        "debugUrl": str(request.url_for("debug_stream_camera", cameraId=camera_id)),
        "contentType": "multipart/x-mixed-replace; boundary=frame",
        "notes": "Use url/path for clean live stream. Use overlayUrl or debugUrl only for parametrage/debug overlays. Do not expose camera RTSP credentials to browser clients.",
    }
