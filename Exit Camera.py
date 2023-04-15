import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime, timedelta
from pymongo import MongoClient
client = MongoClient(
    "mongodb+srv://zaka:1234@cluster0.uyde97r.mongodb.net/?retryWrites=true&w=majority")
# client = MongoClient()
collection = client.detected_faces.appearance

path = 'img'
images = []
className = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])


def encodeImg(imgs):
    encodeList = []
    for img in imgs:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeFrame = face_recognition.face_encodings(frame)[0]
        encodeList.append(encodeFrame)
    return encodeList


EncImg = encodeImg(images)

cap = cv2.VideoCapture(1)
while True:
    _, frame = cap.read()
    FrameResize = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    facesLoc = face_recognition.face_locations(FrameResize)
    encodeImgs = face_recognition.face_encodings(FrameResize, facesLoc)

    for faceLoc, encodeImg in zip(facesLoc, encodeImgs):
        results = face_recognition.compare_faces(EncImg, encodeImg, 0.6)
        face_Dis = face_recognition.face_distance(EncImg, encodeImg)

        cv2.rectangle(frame, (faceLoc[3]*2, faceLoc[0]*2),
                      (faceLoc[1]*2, faceLoc[2]*2), (25, 155, 12), 1)
        index = np.argmin(face_Dis)
        date = datetime.now().replace(second=0, microsecond=0)

        if results[index]:
            name = className[index]
            print(name)
            # insert data in database
            # before verification
            gt = date-timedelta(minutes=2)
            lt = date+timedelta(minutes=2)
            a = collection.find_one({"id": name}, sort=[('checkin', -1)])
            # {'$and': [{"checkin": {'$lte': lt}}, {"checkin": {'$gte': gt}}], "id": name})
            if (a['checkout'] == 'null'):
                collection.update_one(
                    {"_id": a['_id']},
                    {'$set': {
                        "checkout": date
                    }})
            else:
                pass
        else:
            a = collection.find_one({"id": "Unkown"}, sort=[('checkin', -1)])
            if (a['checkout'] == 'null'):
                collection.update_one(
                    {"_id": a['_id']},
                    {'$set': {
                        "checkout": date
                    }})
            else:
                pass
