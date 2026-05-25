import argparse
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

import cv2
import httpx


def load_env(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_rtsp_url(rtsp_url):
    parsed = urlsplit(rtsp_url)
    if parsed.scheme != "rtsp":
        raise ValueError("RTSP_URL must start with rtsp://")

    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    host = parsed.hostname or ""
    channel = extract_channel(parsed.path)

    if not host:
        raise ValueError("Could not parse host from RTSP URL")
    if not username or not password:
        raise ValueError("RTSP URL must contain username and password")

    return {
        "username": username,
        "password": password,
        "host": host,
        "rtsp_port": parsed.port or 554,
        "channel": channel,
    }


def extract_channel(path):
    match = re.search(r"/Streaming/Channels/(\d+)", path)
    if match:
        return match.group(1)
    match = re.search(r"/Streaming/tracks/(\d+)", path)
    if match:
        return match.group(1)
    return "101"


def utc_text(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rtsp_time(value):
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def local_datetime(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def isapi_search_recordings(base_url, username, password, channel, start, end):
    url = f"{base_url}/ISAPI/ContentMgmt/search"
    search_id = str(uuid.uuid4())
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<CMSearchDescription>
  <searchID>{search_id}</searchID>
  <trackList>
    <trackID>{channel}</trackID>
  </trackList>
  <timeSpanList>
    <timeSpan>
      <startTime>{utc_text(start)}</startTime>
      <endTime>{utc_text(end)}</endTime>
    </timeSpan>
  </timeSpanList>
  <maxResults>20</maxResults>
  <searchResultPostion>0</searchResultPostion>
  <metadataList>
    <metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor>
  </metadataList>
</CMSearchDescription>"""

    headers = {"Content-Type": "application/xml", "Accept": "application/xml"}
    auth = httpx.DigestAuth(username, password)
    with httpx.Client(timeout=20.0, verify=False) as client:
        response = client.post(url, content=body.encode("utf-8"), headers=headers, auth=auth)
        if response.status_code == 401:
            response = client.post(url, content=body.encode("utf-8"), headers=headers, auth=(username, password))
        response.raise_for_status()
        return response.text


def parse_search_response(xml_text):
    root = ElementTree.fromstring(xml_text)
    matches = []

    for item in root.iter():
        tag = item.tag.split("}", 1)[-1]
        if tag != "searchMatchItem":
            continue

        data = {}
        for child in item.iter():
            child_tag = child.tag.split("}", 1)[-1]
            if child.text and child_tag in {"trackID", "startTime", "endTime", "playbackURI"}:
                data[child_tag] = child.text.strip()
        if data:
            matches.append(data)

    return matches


def playback_url(host, rtsp_port, username, password, channel, start, end):
    safe_user = quote(username, safe="")
    safe_pass = quote(password, safe="")
    
    return (
        f"rtsp://{safe_user}:{safe_pass}@{host}:{rtsp_port}/Streaming/tracks/{channel}"
        f"?starttime={rtsp_time(start)}&endtime={rtsp_time(end)}"
    )


def test_playback_rtsp(url, timeout_seconds=10):
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    started = datetime.now()
    try:
        while (datetime.now() - started).total_seconds() < timeout_seconds:
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                return True, width, height
        return False, 0, 0
    finally:
        capture.release()


def main():
    load_env()

    parser = argparse.ArgumentParser(description="Check Hikvision/NVR recording history access.")
    parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", ""))
    parser.add_argument("--channel", default="")
    parser.add_argument("--http-scheme", default="http", choices=["http", "https"])
    parser.add_argument("--minutes-back", type=int, default=60)
    parser.add_argument("--clip-seconds", type=int, default=60)
    parser.add_argument("--start", help="UTC start time, format YYYY-MM-DDTHH:MM:SS")
    parser.add_argument("--end", help="UTC end time, format YYYY-MM-DDTHH:MM:SS")
    args = parser.parse_args()

    if not args.rtsp_url:
        raise SystemExit("Missing RTSP URL. Put RTSP_URL in .env or pass --rtsp-url.")

    info = parse_rtsp_url(args.rtsp_url)
    channel = args.channel or info["channel"]

    if args.start and args.end:
        start = local_datetime(args.start)
        end = local_datetime(args.end)
    else:
        end = datetime.now(timezone.utc) - timedelta(minutes=args.minutes_back)
        start = end - timedelta(seconds=args.clip_seconds)

    base_url = f"{args.http_scheme}://{info['host']}"
    print(f"Device: {base_url}")
    print(f"Channel/track: {channel}")
    print(f"Search window UTC: {utc_text(start)} -> {utc_text(end)}")

    print("\n1) Checking ISAPI recording search...")
    try:
        xml_text = isapi_search_recordings(
            base_url=base_url,
            username=info["username"],
            password=info["password"],
            channel=channel,
            start=start,
            end=end,
        )
        matches = parse_search_response(xml_text)
        print(f"ISAPI OK. Matches found: {len(matches)}")
        for match in matches[:5]:
            print(match)
    except Exception as exc:
        print(f"ISAPI FAILED: {exc}")

    print("\n2) Checking playback RTSP...")
    url = playback_url(
        host=info["host"],
        rtsp_port=info["rtsp_port"],
        username=info["username"],
        password=info["password"],
        channel=channel,
        start=start,
        end=end,
    )
    print(url)
    ok, width, height = test_playback_rtsp(url)
    if ok:
        print(f"Playback RTSP OK. First frame: {width}x{height}")
    else:
        print("Playback RTSP FAILED. The device may not support this URL format, or no recording exists in that time range.")


if __name__ == "__main__":
    main()
