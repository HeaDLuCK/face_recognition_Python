from Live import Live
from config import get_camera_source, load_env

if __name__ == "__main__":
    load_env()
    Camera = Live(get_camera_source("HIKVISION_OUT", 1), "out")
    Camera.main()
