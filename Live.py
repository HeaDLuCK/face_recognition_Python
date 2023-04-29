import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime
from Blogic import Blogic


class Live:
    def __init__(self, camera):
        self.camera = camera
        self.cameraValidation()
        self._cap = cv2.VideoCapture(self.camera)
        self._businesslogic = Blogic()

    def screenShot(self, filename, frame):
        path = f"./unkown personnes/{filename}.png"
        if not os.path.exists(path):
            cv2.imwrite(path, frame)

    def cameraValidation(self):
        if not isinstance(self.camera, int):
            raise ValueError('ERROR: Camera Input must be integer ')

    def searchForFaces(self, frame, resized_frame, faces_location, fulldate, image_directory="img", failed=True):
        encodeImgs = face_recognition.face_encodings(
            resized_frame, faces_location)
        for faceLoc, encodeImg in zip(faces_location, encodeImgs):
            images = self._businesslogic.encodeImages(
                self._businesslogic.collectImages(image_directory))
            results = face_recognition.compare_faces(
                images, encodeImg, 0.6)
            face_Dis = face_recognition.face_distance(images, encodeImg)
            cv2.rectangle(
                frame, (faceLoc[3]*2, faceLoc[0]*2), (faceLoc[1]*2, faceLoc[2]*2), (25, 155, 12), 1)
            try:
                index = np.argmin(face_Dis)
                check = results[index]
            except:
                check = False
            if check:
                perso_id = self._businesslogic.collectImagesName(image_directory)[index]
                if (self.camera == 0):
                    self._businesslogic.checkInInsert(fulldate, perso_id)
                elif (self.camera == 1):
                    self._businesslogic.checkOutInsert(fulldate, perso_id)
            else:
                splittedDate = str(fulldate).split(" ")
                date = splittedDate[0]
                hour = splittedDate[1]
                if (self.camera == 0):
                    if not failed:
                        filename = '_'.join((date.split('-'))) + "T" + \
                            '_'.join(hour.split(':'))
                        self.screenShot(filename, resized_frame)
                        self._businesslogic.checkInInsert(fulldate, filename)
                    else:
                        self.searchForFaces(
                            frame, resized_frame, faces_location, fulldate, "unkown personnes", False)
                elif (self.camera == 1):
                    if not failed:
                        filename = '_'.join((date.split('-'))) + "T" + \
                            '_'.join(hour.split(':'))
                        self.screenShot(filename, resized_frame)
                        self._businesslogic.checkOutExceptionInsert(
                            fulldate, filename)
                    else:
                        self.searchForFaces(
                            frame, resized_frame, faces_location, fulldate, "unkown personnes", False)

    def main(self):
        while True:
            _, frame = self._cap.read()
            FrameResize = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
            facesLoc = face_recognition.face_locations(FrameResize)
            fulldate = datetime.now().replace(second=0, microsecond=0)
            if len(facesLoc) > 0:
                self.searchForFaces(frame, FrameResize, facesLoc, fulldate)

            cv2.imshow("test", frame)
            if cv2.waitKey(1) == ord("x"):
                break


