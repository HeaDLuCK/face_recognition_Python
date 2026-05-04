import time

from flask import Flask, Response, abort

from config import get_camera_source, get_int, load_env
from Live import Live


load_env()

app = Flask(__name__)
cameras = {
    "in": Live(get_camera_source("HIKVISION_IN", 0), "in"),
    "out": Live(get_camera_source("HIKVISION_OUT", 1), "out"),
}


for camera in cameras.values():
    camera.start()


def generate_stream(camera):
    while True:
        frame = camera.getJpegFrame()
        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(1 / max(get_int("STREAM_FPS", 10), 1))


@app.get("/")
def index():
    return """
    <html>
      <head><title>Camera Streams</title></head>
      <body>
        <h1>Camera Streams</h1>
        <h2>In</h2>
        <img src="/stream/in" width="640">
        <h2>Out</h2>
        <img src="/stream/out" width="640">
      </body>
    </html>
    """


@app.get("/stream/<direction>")
def stream(direction):
    camera = cameras.get(direction)
    if camera is None:
        abort(404)

    return Response(
        generate_stream(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
def health():
    return {"status": "ok", "cameras": list(cameras.keys())}


if __name__ == "__main__":
    host = "0.0.0.0"
    port = get_int("APP_PORT", 5000)
    app.run(host=host, port=port, threaded=True)
