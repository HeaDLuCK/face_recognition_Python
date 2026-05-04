import queue
import threading
import numpy as np
import face_recognition
import cv2
import os
import time
from datetime import datetime, timedelta
from Blogic import Blogic
from config import get_float, get_int


KNOWN_FACES_PATH = "img"
UNKNOWN_FACES_PATH = "unkown personnes"
UNKNOWN_RECORDINGS_PATH = "unknown_recordings"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class Live:
    
    

    def __init__(self, camera, direction=None):
        self.camera = camera
        self.direction = direction or self._defaultDirection(camera)
        self._cap = cv2.VideoCapture(self.camera)
        self._businesslogic = Blogic()
        self.encode_known_images=[]
        self.encode_unknown_images=[]
        self.images_known_names=[]
        self.images_unknown_names=[]
        self.match_tolerance = get_float("FACE_MATCH_TOLERANCE", 0.5)
        self.unknown_tolerance = get_float("UNKNOWN_FACE_TOLERANCE", 0.5)
        self.detection_cooldown = timedelta(minutes=get_int("DETECTION_COOLDOWN_MINUTES", 2))
        self.unknown_recording_seconds = get_int("UNKNOWN_RECORDING_SECONDS", 30)
        self.last_detection_times = {}
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        self.recording_writer = None
        self.recording_path = None
        self.recording_until = None

        # run methods:
        self.cameraValidation()
        self.encodeImages()
        self.encodeImages(UNKNOWN_FACES_PATH)

        # start background worker so camera reading does not block on recognition/database work
        self.queue = queue.Queue(maxsize=2)
        self.worker_thread = threading.Thread(target=self.runDaemon, daemon=True)
        self.last_reload_date = None
        self.running = False
        
    # take screenShot of frame
    def screenShot(self, filename, frame):
        path = f"./{UNKNOWN_FACES_PATH}/{filename}.png"
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if not os.path.exists(path):
            cv2.imwrite(path, frame)

    def startUnknownRecording(self):
        os.makedirs(UNKNOWN_RECORDINGS_PATH, exist_ok=True)
        now = datetime.now()

        with self.recording_lock:
            self.recording_until = now + timedelta(seconds=self.unknown_recording_seconds)
            if self.recording_writer is not None:
                return

            filename = f"{self.direction}_{now.strftime('%Y%m%d%H%M%S')}.avi"
            self.recording_path = os.path.join(UNKNOWN_RECORDINGS_PATH, filename)
            print(f"scheduled unknown recording: {self.recording_path}")

    def writeRecordingFrame(self, frame):
        with self.recording_lock:
            if self.recording_until is None or self.recording_path is None:
                return

            if self.recording_writer is None:
                height, width = frame.shape[:2]
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 15
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.recording_writer = cv2.VideoWriter(self.recording_path, fourcc, fps, (width, height))
                print(f"started unknown recording: {self.recording_path}")

            if datetime.now() >= self.recording_until:
                self.recording_writer.release()
                self.recording_writer = None
                self.recording_path = None
                self.recording_until = None
                print("stopped unknown recording")
                return

            self.recording_writer.write(frame)

    def setLatestFrame(self, frame):
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()

    def getJpegFrame(self):
        with self.latest_frame_lock:
            if self.latest_frame is None:
                return None

            frame = self.latest_frame.copy()

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None

        return buffer.tobytes()

    def _defaultDirection(self, camera):
        if camera == 0:
            return "in"
        if camera == 1:
            return "out"
        return "in"

    #validate if input is a local camera index or a stream url
    def cameraValidation(self):
        if not isinstance(self.camera, (int, str)):
            raise ValueError('ERROR: Camera input must be an integer index or stream URL')
        if self.direction not in ("in", "out"):
            raise ValueError('ERROR: Camera direction must be "in" or "out"')


    def encodeImages(self, path=KNOWN_FACES_PATH):
        encodings = []
        names = []

        if not os.path.exists(path):
            os.makedirs(path)

        for img in os.listdir(path):
            if not img.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(path, img)
            frame = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if len(face_encodings) != 1:
                print(f"skipped {image_path}: expected 1 face, found {len(face_encodings)}")
                continue

            encodings.append(face_encodings[0])
            names.append(os.path.splitext(img)[0])

        if path == KNOWN_FACES_PATH:
            self.encode_known_images = encodings
            self.images_known_names = names
            print(f"loaded {len(names)} known face encodings")
        elif path == UNKNOWN_FACES_PATH:
            self.encode_unknown_images = encodings
            self.images_unknown_names = names
            print(f"loaded {len(names)} unknown face encodings")

    def findBestMatch(self, encodedImages, imageNames, face_encoding, tolerance):
        if len(encodedImages) == 0:
            return None, None

        face_distances = face_recognition.face_distance(encodedImages, face_encoding)
        index = np.argmin(face_distances)
        distance = face_distances[index]

        if distance <= tolerance:
            return imageNames[index], distance

        return None, distance

    def shouldInsertDetection(self, perso_id, fulldate):
        key = (self.direction, str(perso_id))
        last_seen = self.last_detection_times.get(key)

        if last_seen and fulldate - last_seen < self.detection_cooldown:
            return False

        self.last_detection_times[key] = fulldate
        return True

    def registerKnownFace(self, perso_id, fulldate):
        if not self.shouldInsertDetection(perso_id, fulldate):
            return

        print(f"matched employee {perso_id}")
        if self.direction == "in":
            self._businesslogic.checkInInsert(fulldate, int(perso_id))
        elif self.direction == "out":
            self._businesslogic.checkOutInsert(fulldate, int(perso_id))

    def registerUnknownFace(self, face_encoding, frame, face_location, fulldate):
        unknown_id, _ = self.findBestMatch(
            self.encode_unknown_images,
            self.images_unknown_names,
            face_encoding,
            self.unknown_tolerance
        )

        if unknown_id:
            return

        self.startUnknownRecording()

        filename = datetime.now().strftime("%Y%m%d%H%M%S%f")
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]

        if face_image.size == 0:
            return

        self.screenShot(filename, face_image)
        self.encodeImages(UNKNOWN_FACES_PATH)

        if self.direction == "in":
            self._businesslogic.checkInInsert(fulldate, filename)
        elif self.direction == "out":
            self._businesslogic.checkOutExceptionInsert(fulldate, filename)


    # looking for faces if they are  in the folder img or unkown personnes
    def searchForFaces(self, 
                       resized_frame, 
                       faces_location,
                       fulldate):
        encodeImgs = face_recognition.face_encodings(
            resized_frame, faces_location)
        
        for encodeImg, face_location in zip(encodeImgs, faces_location):
            perso_id, distance = self.findBestMatch(
                self.encode_known_images,
                self.images_known_names,
                encodeImg,
                self.match_tolerance
            )

            if perso_id:
                self.registerKnownFace(perso_id, fulldate)
            else:
                print(f"unknown face, nearest distance: {distance}")
                self.registerUnknownFace(encodeImg, resized_frame, face_location, fulldate)


    #daemon helper
    def runDaemon(self):
        while True:
            item = self.queue.get()
            if item is None:
                break

            resized_frame, faces_location, fulldate = item
            self.searchForFaces(resized_frame, faces_location, fulldate)

    def queueFaceSearch(self, resized_frame, faces_location, fulldate):
        try:
            self.queue.put_nowait((resized_frame, faces_location, fulldate))
        except queue.Full:
            pass

    def reloadImagesOncePerDay(self):
        today = datetime.now().date()
        if self.last_reload_date == today:
            return

        self.encodeImages()
        self.encodeImages(UNKNOWN_FACES_PATH)
        self.last_reload_date = today

    #start daemon and send data to daemon so the camera wont freeze and data gonna be inserted/updated
    def main(self, show_window=True):
        if not self.worker_thread.is_alive():
            self.worker_thread.start()

        self.running = True
        try:
            while self.running:
                ok, frame = self._cap.read()
                if not ok:
                    print("camera frame not available")
                    time.sleep(1)
                    break

                FrameResize = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
                FrameResize = cv2.cvtColor(FrameResize,cv2.COLOR_BGR2RGB)
                facesLoc = face_recognition.face_locations(FrameResize)
                fulldate = datetime.now().replace(second=0, microsecond=0)
                for faceLoc in facesLoc:
                    cv2.rectangle(
                        frame, (faceLoc[3]*2, faceLoc[0]*2), (faceLoc[1]*2, faceLoc[2]*2), (25, 155, 12), 1)
                if len(facesLoc) > 0:
                    self.queueFaceSearch(FrameResize, facesLoc, fulldate)

                self.writeRecordingFrame(frame)
                self.setLatestFrame(frame)
                    
                # update every day at 00:00
                if datetime.now().strftime("%H:%M")=="00:00":
                    self.reloadImagesOncePerDay()
                
                if show_window:
                    cv2.imshow(f"camera-{self.direction}", frame)
                    if cv2.waitKey(1) == ord("x"):
                        break

        except Exception as e:
            print(f"camera loop stopped: {e}")
        finally:
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            with self.recording_lock:
                if self.recording_writer is not None:
                    self.recording_writer.release()
                    self.recording_writer = None
            cv2.destroyAllWindows()

    def start(self):
        thread = threading.Thread(target=self.main, kwargs={"show_window": False}, daemon=True)
        thread.start()
        return thread

    def stop(self):
        self.running = False
