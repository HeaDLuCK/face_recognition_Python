# face_recognition_Python

## Face verification flow

Known employee images go in `img/`. The file name without extension is used as the employee id, for example `img/12.jpg` becomes employee id `12`.

Camera entry points:

- `Camera1.py`: check-in camera
- `Camera2.py`: check-out camera

The face verification logic is in `Live.py`:

- `encodeImages()` loads and encodes faces from `img/` and `unkown personnes/`
- `findBestMatch()` compares a live face encoding to stored encodings using face distance
- `registerKnownFace()` writes employee check-in/check-out records
- `registerUnknownFace()` saves unknown faces and writes exception records

## Tuning

Edit `.env`:

```env
FACE_MATCH_TOLERANCE=0.5
UNKNOWN_FACE_TOLERANCE=0.5
DETECTION_COOLDOWN_MINUTES=2
```

Lower tolerance is stricter. If the system matches the wrong person, try `0.45`. If it misses real employees too often, try `0.55`.

The cooldown prevents the same face from being inserted every frame while the person is standing in front of the camera.

## Camera streaming

Run the web server:

```powershell
python app.py
```

Open:

- `http://localhost:5000/` to see both streams
- `http://localhost:5000/stream/in` for the check-in camera
- `http://localhost:5000/stream/out` for the check-out camera
- `http://localhost:5000/health` for a quick health check

The stream is MJPEG, so it can be embedded in a browser with:

```html
<img src="http://localhost:5000/stream/in">
```

## Unknown-person recording

When a face does not match a known employee or a previously saved unknown face, the app:

- saves the cropped face image in `unkown personnes/`
- starts recording the camera feed in `unknown_recordings/`
- records for `UNKNOWN_RECORDING_SECONDS`

Configure it in `.env`:

```env
UNKNOWN_RECORDING_SECONDS=30
STREAM_FPS=10
APP_PORT=5000
```

## Customer visit duration

Yes, it is possible to detect when a customer enters a store and estimate how long they stay, but it should not be built with face recognition as the main tool.

The better approach is person detection and tracking:

- detect people with a model like YOLO
- track each person with a tracker like ByteTrack or DeepSORT
- define an entrance line or store zone
- start a timer when a tracked person crosses into the zone
- stop the timer when the same tracked person leaves

This can measure customer dwell time without identifying the person's face, which is usually better for privacy and reliability.
