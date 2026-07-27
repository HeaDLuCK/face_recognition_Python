# AI Camera Service

FastAPI microservice that runs AI processing for ERP-managed camera systems.

ERP is the source of truth. This service does not create or manage cameras, employees, face images, zones, camera capabilities, or attendance rules as master data. It fetches them from ERP, generates local face embeddings, processes RTSP streams, stores runtime artifacts in MongoDB, saves snapshots locally, and sends events back to ERP.

## Current Scope

Implemented now:

- ERP sync for cameras, employees, employee face images, and attendance rules.
- Local cached employee face embeddings with InsightFace.
- RTSP frame reading with OpenCV.
- Capability-gated `FACE_RECOGNITION`.
- Capability-gated YOLO `PLATE_RECOGNITION` and `FIRE_DETECTION`.
- Fair round-robin face inference across running cameras.
- Optional motion-gated person tracking with separate recognition jobs per person.
- Automatic Hikvision history recovery for unresolved attendance tracks.
- Zone filtering for detections when ERP provides zones.
- Attendance rules:
  - camera direction `IN` / `OUT`
  - duplicate cooldown
  - recognition confidence threshold
- MongoDB runtime storage.
- ERP event delivery through `POST {ERP_BASE_URL}/api/ai/events`.

Architecture placeholders exist for future modules:

- `OBJECT_COUNTING`
- `PERSON_COUNTING`
- `SMOKE_DETECTION`
- `SUSPICIOUS_BEHAVIOR`
- `POSTURE_DETECTION`

These modules are not implemented yet.

## Runtime MongoDB Collections

The service writes runtime data only:

- `cached_embeddings`
- `attendance_detections`
- `camera_events`
- `alert_events`
- `snapshot_metadata`
- `service_logs`
- `attendance_recovery_jobs`

Every runtime document includes `tenantId`.

## ERP Endpoints Used

```text
GET  {ERP_BASE_URL}/api/ai/cameras
GET  {ERP_BASE_URL}/api/ai/employees?tenantId=COMPANY_01
GET  {ERP_BASE_URL}/api/ai/attendance-rules?tenantId=COMPANY_01
POST {ERP_BASE_URL}/api/ai/events
```

Expected camera payload:

```json
{
  "tenantId": "COMPANY_01",
  "cameraId": "CAM_01",
  "name": "Main Entrance",
  "rtspUrl": "rtsp://user:pass@192.168.1.50:554/stream1",
  "enabled": true,
  "direction": "IN",
  "capabilities": ["FACE_RECOGNITION"],
  "zones": [
    {
      "zoneId": "ZONE_01",
      "name": "Door Area",
      "x": 100,
      "y": 50,
      "width": 500,
      "height": 700
    }
  ]
}
```

Expected attendance rules payload:

```json
{
  "tenantId": "COMPANY_01",
  "recognitionThreshold": 0.55,
  "duplicateCooldownSeconds": 60,
  "personTrackingEnabled": true,
  "historyRecoveryEnabled": true,
  "saveUnknownFaces": true,
  "sendUnknownFaceAlert": false
}
```

Events sent back to ERP follow this shape:

```json
{
  "tenantId": "COMPANY_01",
  "cameraId": "CAM_01",
  "eventType": "ATTENDANCE_IN",
  "employeeId": "EMP_001",
  "confidence": 0.92,
  "snapshotPath": "snapshots/COMPANY_01/CAM_01/xxx.jpg",
  "timestamp": "2026-05-05T12:30:10"
}
```

## Setup

```bash
cd ai_camera_service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```text
ERP_BASE_URL=https://your-erp.example.com
ERP_API_TOKEN=your-token
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=ai_camera_service
SNAPSHOT_DIR=snapshots
DEFAULT_RECOGNITION_THRESHOLD=0.55
DEFAULT_DUPLICATE_COOLDOWN_SECONDS=60
```

Run:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs:

```text
http://localhost:8000/docs
```

InsightFace downloads its configured model on first use. For GPU inference, install a compatible ONNX Runtime GPU build, set `INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`, and set `INSIGHTFACE_CTX_ID=0`.

## Control API

```text
GET  /health
GET  /api/status

POST /api/sync/all
POST /api/sync/cameras
POST /api/sync/employees
POST /api/sync/rules

POST /api/cameras/{cameraId}/start
POST /api/cameras/{cameraId}/stop
POST /api/cameras/start-all
POST /api/cameras/stop-all

GET  /api/events?tenantId=COMPANY_01
GET  /api/attendance?tenantId=COMPANY_01
POST /api/test/recognize-image

GET  /api/cameras/grid
GET  /api/cameras/{cameraId}/stream
GET  /api/cameras/{cameraId}/stream-flow
GET  /api/cameras/stream-flows
POST /api/cameras/discover-channels

GET  /api/recovery-jobs?etsAuth=COMPANY_01
POST /api/recovery-jobs/{recoveryJobId}/retry
```

## Person Tracking And Attendance Recovery

Person tracking is disabled by default so existing deployments keep their
current face-recognition behavior. To enable it:

1. Put a lightweight Ultralytics COCO person model at
   `app/tracking/model/person_yolo.pt`. The model must contain the standard
   `person` class (`class 0`).
2. Set `PERSON_TRACKING_ENABLED=true`.
3. Keep `personTrackingEnabled=true` and `historyRecoveryEnabled=true` in the
   tenant attendance rules.
4. Restart the service and cameras.

For development, Ultralytics can download a nano COCO model once, after which
you can rename/copy it to the configured path:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Validate the person model before enabling all cameras:

```bash
python check_yolo_person_image.py path/to/test.jpg --save-debug person_debug.jpg
```

When enabled, empty cameras are gated by cheap motion checks. Detected people
receive temporary per-camera track IDs, and face jobs are kept separately per
track. A track that ends without recognition creates `PERSON_UNIDENTIFIED` and
an `attendance_recovery_jobs` document. The recovery worker waits for live face
AI to become idle, exports the small Hikvision history window, and records
matches with `metadata.source=HISTORY_RECOVERY`.

Verify the runtime state:

```text
GET /api/status
```

Check `personDetector.modelAvailable`, each camera's
`runtime.personTrackingEnabled`, `activePersonTracks`, and
`personTracksUnresolved`.

## Run Modes

### Development With A USB Camera

For local testing, leave `ERP_BASE_URL` empty in `.env` and use:

```text
ENVIRONMENT=development
ERP_BASE_URL=
CAMERA_SOURCE_MODE=usb
USB_CAMERA_INDEX=0
DEV_TENANT_ID=DEV_COMPANY
DEV_CAMERA_ID=USB_CAM_01
STREAM_FPS=20
STREAM_JPEG_QUALITY=80
SHOW_DEV_FPS=true
SHOW_DEV_DETECTIONS=true
```

Start MongoDB, then run the API:

```bash
cd ai_camera_service
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Create the local development camera config:

```bash
curl -X POST http://localhost:8000/api/sync/cameras
curl -X POST http://localhost:8000/api/sync/rules
```

Start the USB camera:

```bash
curl -X POST http://localhost:8000/api/cameras/USB_CAM_01/start
```

Check status:

```bash
curl http://localhost:8000/api/status
```

Open the camera grid in your browser:

```text
http://localhost:8000/api/cameras/grid
```

Open one camera stream directly:

```text
http://localhost:8000/api/cameras/USB_CAM_01/stream
```

Calibrate watched motion zones by clicking points on the stream:

```text
http://localhost:8000/api/cameras/USB_CAM_01/calibrate
```

Stop the USB camera:

```bash
curl -X POST http://localhost:8000/api/cameras/USB_CAM_01/stop
```

If your webcam is not camera `0`, try `USB_CAMERA_INDEX=1`.

If the browser preview still feels slow, increase `CAMERA_FRAME_SKIP` so AI recognition runs less often, for example:

```text
CAMERA_FRAME_SKIP=15
```

The video preview can stay fast because streaming is now decoupled from recognition work. `SHOW_DEV_FPS=true` draws the current development FPS on the stream. `SHOW_DEV_DETECTIONS=true` draws a triangle and box when face detection finds a face.

### Production With IP/RTSP Cameras

In production, set ERP and RTSP mode:

```text
ENVIRONMENT=production
ERP_BASE_URL=https://your-erp.example.com
ERP_API_TOKEN=your-token
CAMERA_SOURCE_MODE=rtsp
```

ERP camera configs should contain RTSP URLs, for example:

```text
rtsp://username:password@192.168.1.50:554/Streaming/Channels/101
```

Then start the service and sync from ERP:

```bash
curl -X POST http://localhost:8000/api/sync/all
curl -X POST http://localhost:8000/api/cameras/start-all
```

### ERP Config Option For USB Testing

If you want ERP to send a USB camera during development, set the camera source as either:

```json
{
  "rtspUrl": "usb://0"
}
```

or:

```json
{
  "rtspUrl": "0"
}
```

With `CAMERA_SOURCE_MODE=auto`, the service treats those as local USB camera index `0`. Normal `rtsp://...` values are treated as IP camera streams.

### Discover Hikvision Channels For ERP

ERP can ask this service to test Hikvision/NVR channels and return only the working RTSP channels.

```powershell
Invoke-RestMethod -Method POST http://localhost:8000/api/cameras/discover-channels `
  -ContentType "application/json" `
  -Body '{
    "tenantId": "COMPANY_01",
    "rtspUrl": "rtsp://admin:password@192.168.100.5:554/Streaming/Channels/101",
    "maxCamera": 16,
    "timeoutSeconds": 4
  }'
```

Response:

```json
{
  "tenantId": "COMPANY_01",
  "count": 3,
  "workingChannels": [
    {
      "channel": "101",
      "rtspUrl": "rtsp://admin:password@192.168.100.5:554/Streaming/Channels/101",
      "width": 1920,
      "height": 1080
    }
  ],
  "rtspChannels": ["101", "201", "301"],
  "envValue": "101,201,301"
}
```

If ERP wants stream information for browser/cloud display, call:

```text
GET /api/cameras/{cameraId}/stream-flow
GET /api/cameras/stream-flows
```

These return the application stream endpoint, not the raw RTSP URL.

## Typical Flow

Sync all ERP data:

```bash
curl -X POST http://localhost:8000/api/sync/all
```

Start one camera:

```bash
curl -X POST http://localhost:8000/api/cameras/CAM_01/start
```

Start all enabled synced cameras:

```bash
curl -X POST http://localhost:8000/api/cameras/start-all
```

Check status:

```bash
curl http://localhost:8000/api/status
```

Test recognition with cached ERP embeddings:

```bash
curl -X POST http://localhost:8000/api/test/recognize-image ^
  -F tenantId=COMPANY_01 ^
  -F file=@test.jpg
```

## Notes

- ERP employees must include face image references or base64 image content in `faceImages`.
- Recognition compares detected embeddings only against `cached_embeddings` for the same `tenantId`.
- If zones are present on a camera, face detections outside all zones are ignored.
- `BIDIRECTIONAL` cameras can produce recognition events, but attendance logs are only generated for `IN` or `OUT` cameras.

## Event Markers, History, And Motion Zones

The service does not keep a full-frame rolling video buffer in Python. Face, plate, fire, and motion events store a timestamp plus playback start/end metadata. The ERP can request that time window from Hikvision through `POST /api/cameras/{cameraId}/history-clip`; the generated temporary MP4 is deleted after the response is delivered.

Configure in `.env`:

```text
EVENT_CLIP_COOLDOWN_SECONDS=30
MOTION_ZONES=door:100,120|420,120|420,360|100,360
MOTION_CHECK_FRAME_SKIP=5
MOTION_PIXEL_THRESHOLD=35
MOTION_AREA_RATIO=0.02
SHOW_MOTION_ZONES=true
```

## Performance Controls

Defaults are conservative for computers that run several cameras on CPU:

```text
OPENCV_NUM_THREADS=1
INSIGHTFACE_DET_SIZE=640
FACE_CANDIDATE_BUFFER_SIZE=4
FACE_CANDIDATE_WINDOW_SECONDS=0.5
UNKNOWN_FACE_CACHE_MAX_ENTRIES=1000
UNKNOWN_FACE_CROP_CACHE_MAX_ENTRIES=500
UNKNOWN_FACE_DB_MATCH_LIMIT=500
```

Face inference is serialized by one shared round-robin scheduler. Every camera keeps one latest pending candidate, so slow hardware does not build a stale frame queue. Increase `INSIGHTFACE_DET_SIZE` only when hardware capacity and face distance require it.

`MOTION_ZONES` supports polygons. Use `;` between zones and `|` between points:

```text
MOTION_ZONES=door:100,120|420,120|420,360|100,360;shelf:600,150|900,150|900,500|600,500
```

To draw a zone, sync and start the camera, then open:

```text
http://localhost:8000/api/cameras/RTSP_CAM_01/calibrate
```

Click around the area to watch, copy the generated `MOTION_ZONES=...` value into `.env`, then restart the API and start the camera again.

## Push Live Streams To Cloud

The office service can push each running camera stream to a cloud WebSocket endpoint. This keeps RTSP cameras private on the office network; only the local service connects outward to your cloud.

Configure `.env`:

```text
CLOUD_STREAM_WS_URL=wss://your-cloud.example.com/ws/camera-ingest
CLOUD_STREAM_TOKEN=your-shared-secret
CLOUD_STREAM_FPS=10
CLOUD_STREAM_RECONNECT_SECONDS=5
```

Leave `CLOUD_STREAM_WS_URL` empty to disable cloud pushing.

When a camera starts, the service connects to the cloud endpoint and sends one JSON text message:

```json
{
  "type": "camera_start",
  "tenantId": "DEV_COMPANY",
  "cameraId": "RTSP_CAM_01",
  "cameraName": "Local RTSP Camera 01",
  "token": "your-shared-secret",
  "timestamp": "2026-05-20T12:00:00"
}
```

After that, it sends binary WebSocket messages. Each binary message is one JPEG frame.

Your cloud system should:

- accept the WebSocket connection
- validate `token`
- map the connection to `tenantId` and `cameraId`
- store the latest JPEG frame for that camera
- serve it to browser users as MJPEG, WebSocket frames, or convert it to HLS/WebRTC
