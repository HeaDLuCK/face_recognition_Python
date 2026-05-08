# AI Camera Service

FastAPI microservice that runs AI processing for ERP-managed camera systems.

ERP is the source of truth. This service does not create or manage cameras, employees, face images, zones, camera capabilities, or attendance rules as master data. It fetches them from ERP, generates local face embeddings, processes RTSP streams, stores runtime artifacts in MongoDB, saves snapshots locally, and sends events back to ERP.

## Current Scope

Implemented now:

- ERP sync for cameras, employees, employee face images, and attendance rules.
- Local cached employee face embeddings with InsightFace.
- RTSP frame reading with OpenCV.
- Capability-gated `FACE_RECOGNITION`.
- Zone filtering for detections when ERP provides zones.
- Attendance rules:
  - camera direction `IN` / `OUT`
  - duplicate cooldown
  - recognition confidence threshold
- MongoDB runtime storage.
- ERP event delivery through `POST {ERP_BASE_URL}/api/ai/events`.

Architecture placeholders exist for future modules:

- `PLATE_RECOGNITION`
- `OBJECT_COUNTING`
- `PERSON_COUNTING`
- `SMOKE_DETECTION`
- `FIRE_DETECTION`
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
```

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
