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

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    FrameResize = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    facesLoc = face_recognition.face_locations(FrameResize)
    encodeImgs = face_recognition.face_encodings(FrameResize, facesLoc)
    print(facesLoc)

    # fulldate = datetime.now().replace(second=0, microsecond=0)
    # splittedDate = str(fulldate).split(" ")
    # date = splittedDate[0]
    # hour = splittedDate[1]
    # hourplusOne = str(fulldate+timedelta(minutes=1)).split(" ")[1]
    # hourplusTwo = str(fulldate+timedelta(minutes=2)).split(" ")[1]
    # for faceLoc, encodeImg in zip(facesLoc, encodeImgs):
    #     results = face_recognition.compare_faces(EncImg, encodeImg, 0.6)
    #     face_Dis = face_recognition.face_distance(EncImg, encodeImg)

    #     cv2.rectangle(frame, (faceLoc[3]*2, faceLoc[0]*2),
    #                   (faceLoc[1]*2, faceLoc[2]*2), (25, 155, 12), 1)
    #     index = np.argmin(face_Dis)
        # if results[index]:
        #     perso_id = className[index]
        #     print(perso_id)
            # dateCheck = collection.find_one({"date": date})
            # if (dateCheck):
            #     a = collection.find_one(
            #         {"presence.checkIn": {'$in': [hour, hourplusOne, hourplusTwo]}, "date": date, "presence.perso_id": perso_id})
            #     if (a):
            #         pass
            #     else:
            #         collection.update_one(
            #             {"_id": dateCheck['_id']},
            #             {"$push": {"presence": {"perso_id": perso_id, "checkIn": hour,
            #                        "checkOut": 'null'}}})
            # else:

            #     collection.insert_one(
            #         {"date": date, "presence": [
            #             {"perso_id": perso_id, "checkIn": hour, "checkOut": 'null'}
            #         ]})

        # else:
        #     print("Unkown")
            # a = collection.find_one({"date": date})
            # filename = '_'.join((date.split('-'))) + "T" + \
            #     '_'.join(hour.split(':'))
            # path = f"./unkown personnes/{filename}.png"
            # cv2.imwrite(path, FrameResize)
            # if (a):
            #     check = collection.find_one(
            #         {"presence.checkIn": {'$in': [hour, hourplusOne, hourplusTwo]}, "date": date, "presence.perso_id": filename})
            #     if (check):
            #         pass
            #     else:
            #         collection.update_one(
            #             {"_id": a['_id']},
            #             {'$push': {
            #                 "presence": {"perso_id": filename, "checkIn": hour,
            #                              "checkOut": 'null'}
            #             }})
            # else:
            #     collection.insert_one(
            #         {"date": date, "presence": [
            #             {"perso_id": filename, "checkIn": hour, "checkOut": 'null'}
            #         ]})

    cv2.imshow("test", frame)
    if cv2.waitKey(1) == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()
