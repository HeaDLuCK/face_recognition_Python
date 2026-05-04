import os


def load_env(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def get_camera_source(prefix, fallback):
    host = os.getenv(f"{prefix}_IP")
    username = os.getenv(f"{prefix}_USERNAME")
    password = os.getenv(f"{prefix}_PASSWORD")
    port = os.getenv(f"{prefix}_RTSP_PORT", "554")
    channel = os.getenv(f"{prefix}_CHANNEL", "101")

    if not host:
        return os.getenv(f"{prefix}_RTSP", fallback)

    if not username or not password:
        raise ValueError(f"{prefix}_USERNAME and {prefix}_PASSWORD are required when {prefix}_IP is set")

    return f"rtsp://{username}:{password}@{host}:{port}/Streaming/Channels/{channel}"


def get_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def get_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)
