from html import escape
import asyncio
import time
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

import cv2
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class ChannelDiscoveryRequest(BaseModel):
    tenantId: str | None = None
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
    channels: list[str] | None = None
    maxCamera: int = Field(default=16, ge=1, le=64)
    timeoutSeconds: int = Field(default=2, ge=1, le=20)
    discoveryConcurrency: int = Field(default=8, ge=1, le=32)
    validateRtsp: bool = True


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

    cameras = _group_discovered_cameras(results, payload)

    return {
        "tenantId": payload.tenantId,
        "cameraDeviceId": payload.cameraDeviceId,
        "discoverySource": discovery_source,
        "rtspValidated": payload.validateRtsp or discovery_source == "rtsp_probe",
        "count": len(cameras),
        "cameras": cameras,
        "workingChannels": results,
        "rtspChannels": [item["channel"] for item in results],
        "envValue": ",".join(item["channel"] for item in results),
    }


async def _discover_channel(
    payload: ChannelDiscoveryRequest,
    base_url: str,
    channel: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    rtsp_url = f"{base_url}/{channel}"
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
    rtsp_url = f"{base_url}/{channel}"
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

    username = quote(payload.username, safe="")
    password = quote(payload.password, safe="")
    return f"rtsp://{username}:{password}@{payload.ip}:{payload.rtspPort}/Streaming/Channels"


def _discover_channels_from_isapi(payload: ChannelDiscoveryRequest) -> list[str]:
    connection = _device_connection(payload)
    if connection is None:
        return []

    scheme, host, username, password = connection
    url = f"{scheme}://{host}/ISAPI/Streaming/channels"
    try:
        with httpx.Client(timeout=payload.timeoutSeconds, verify=False) as client:
            response = client.get(url, auth=httpx.DigestAuth(username, password))
            if response.status_code == 401:
                response = client.get(url, auth=(username, password))
            response.raise_for_status()
    except Exception:
        return []

    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError:
        return []

    channels = []
    for element in root.iter():
        tag = element.tag.split("}", 1)[-1]
        if tag == "id" and element.text:
            channel = element.text.strip()
            if channel and channel not in channels:
                channels.append(channel)
    return channels


def _device_connection(payload: ChannelDiscoveryRequest) -> tuple[str, str, str, str] | None:
    if payload.ip and payload.username and payload.password:
        return "http", payload.ip, payload.username, payload.password

    if not payload.rtspUrl:
        return None

    parsed = urlsplit(payload.rtspUrl)
    if not parsed.hostname or not parsed.username or not parsed.password:
        return None

    return "http", parsed.hostname, unquote(parsed.username), unquote(parsed.password)


def _default_hikvision_channels(max_camera: int) -> list[str]:
    channels = []
    for camera_number in range(1, max_camera + 1):
        channels.append(f"{camera_number}01")
        channels.append(f"{camera_number}02")
    return channels


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
        while time.time() - started_at < timeout_seconds:
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                return True, int(width), int(height)
            time.sleep(0.1)
        return False, 0, 0
    finally:
        capture.release()


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
