import multiprocessing
import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime
from Blogic import Blogic


class Live:
    
    

    def __init__(self, camera):
        self.camera = camera
        self._cap = cv2.VideoCapture(self.camera)
        self._businesslogic = Blogic()
        self.encode_known_images=[]
        self.encode_unknown_images=[]
        self.images_known_names=[]
        self.images_unknown_names=[]

        # run methods:
        self.cameraValidation()
        self.encodeImages()
        self.encodeImages("unkown personnes")

        # start daemon that check faces and insert data
        self.queue = multiprocessing.Queue() 
        self.daemon_process = multiprocessing.Process(target=self.runDaemon)
        self.daemon_process.daemon = True
        
    # take screenShot of frame
    def screenShot(self, filename, frame):
        path = f"./unkown personnes/{filename}.png"
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if not os.path.exists(path):
            cv2.imwrite(path, frame)

    #validate if input is integer
    def cameraValidation(self):
        if not isinstance(self.camera, int):
            raise ValueError('ERROR: Camera Input must be integer ')


    #encode images
    def encodeImages(self, path="img"):
        if(path=="img"):
            for img in os.listdir(path):
                if img not in self.images_known_names:
                    frame = face_recognition.load_image_file(f'{path}/{img}')
                    encodeFrame = face_recognition.face_encodings(frame)[0]
                    self.encode_known_images.append(encodeFrame)
                    self.images_known_names.append(os.path.splitext(img)[0])
            print('encode images ended :D')
            print(self.images_known_names)
        elif path=="unkown personnes":
            self.encode_unknown_images=[]
            for img in os.listdir(path):
                if img not in self.images_unknown_names:
                    frame = face_recognition.load_image_file(f'{path}/{img}')
                    encodeFrame = face_recognition.face_encodings(frame)[0]
                    self.encode_unknown_images.append(encodeFrame)
                    self.images_unknown_names.append(os.path.splitext(img)[0])
            print('encode unkown images ended :D')
            print(self.images_unknown_names)


    # looking for faces if they are  in the folder img or unkown personnes
    def searchForFaces(self, 
                       resized_frame, 
                       faces_location,
                       encodedImages,
                       imageNames, 
                       fulldate, 
                       failed=True):
        encodeImgs = face_recognition.face_encodings(
            resized_frame, faces_location)
        
        for encodeImg in encodeImgs:
            results = face_recognition.compare_faces(
                encodedImages, encodeImg, 0.6)
            face_Dis = face_recognition.face_distance(encodedImages, encodeImg)
            try:
                index = np.argmin(face_Dis)
                check = results[index]
            except:
                check = False
            if check:
                perso_id = imageNames[index]
                print(perso_id)
                if (self.camera == 0):
                    
                    self._businesslogic.checkInInsert(fulldate, int(perso_id))
                elif (self.camera == 1):
                    self._businesslogic.checkOutInsert(fulldate, int(perso_id))
            else:
                date,time=str(fulldate).split(" ")
                if (self.camera == 0):
                    if not failed:
                        #filename looks like 20141010141200 = 2014-10-10 T 14:12:00
                        filename = ''.join((date.split('-')))  + \
                            ''.join(time.split(':'))
                        self.screenShot(filename, resized_frame)
                        print("i took screenshoot")
                        self._businesslogic.checkInInsert(fulldate, filename)
                    else:
                         self.queue.put(
                            (resized_frame, faces_location, self.encode_unknown_images, self.images_unknown_names, fulldate, False))


                elif (self.camera == 1):
                    if not failed:
                        #filename looks like 20141010141200 = 2014-10-10 T 14:12:00
                        filename = ''.join((date.split('-'))) + \
                            ''.join(time.split(':'))
                        self.screenShot(filename, resized_frame)
                        self._businesslogic.checkOutExceptionInsert(
                            fulldate, filename)
                    else:
                         self.queue.put(
                            (resized_frame, faces_location, self.encode_unknown_images, self.images_unknown_names, fulldate, False))


    #daemon helper
    def runDaemon(self):
        while True:
            resized_frame, faces_location, encodedImages, imageNames, fulldate, failed = self.queue.get()
            self.searchForFaces(resized_frame, faces_location, encodedImages, imageNames, fulldate, failed)

    #start daemon and send data to daemon so the camera wont freeze and data gonna be inserted/updated
    def main(self):
        if self.daemon_process.is_alive():
            self.daemon_process.terminate()
        self.daemon_process.start()

        
        try:
            while True:
                _, frame = self._cap.read()
                FrameResize = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
                FrameResize = cv2.cvtColor(FrameResize,cv2.COLOR_RGB2BGR)
                facesLoc = face_recognition.face_locations(FrameResize)
                fulldate = datetime.now().replace(second=0, microsecond=0)
                for faceLoc in facesLoc:
                    cv2.rectangle(
                        frame, (faceLoc[3]*2, faceLoc[0]*2), (faceLoc[1]*2, faceLoc[2]*2), (25, 155, 12), 1)
                if len(facesLoc) > 0:
                    self.queue.put((
                                    FrameResize,
                                    facesLoc, 
                                    self.encode_known_images,
                                    self.images_known_names,
                                    fulldate,
                                    True))
                    
                # update every day at 00:00
                if datetime.now().strftime("%H:%M")=="00:00":
                    self.encodeImages()
                    self.encodeImages("unkown personnes")
                
                cv2.imshow("test", frame)
                if cv2.waitKey(1) == ord("x"):
                    break

            self.daemon_process.terminate()
        except Exception as e:
            print("no camera detected")