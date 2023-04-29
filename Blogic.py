from datetime import timedelta
import os
import cv2
from pymongo import MongoClient
import face_recognition


class Blogic:
    def __init__(self):
        self.__database = MongoClient(
            "mongodb+srv://zaka:1234@cluster0.uyde97r.mongodb.net/?retryWrites=true&w=majority")
        self.collection = self._Blogic__database.detected_faces.appearance

    def collectImagesName(self, path):
        className = []
        myList = os.listdir(path)
        for cl in myList:
            className.append(os.path.splitext(cl)[0])
        return className

    def collectImages(self, path):
        myList = os.listdir(path)
        images = []
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
        return images

    def encodeImages(self, images):
        encodedImages = []
        for img in images:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodeFrame = face_recognition.face_encodings(frame)[0]
            encodedImages.append(encodeFrame)
        return encodedImages

    def checkInInsert(self, fulldate, perso_id):
        splittedDate = str(fulldate).split(" ")
        date = splittedDate[0]
        hour = splittedDate[1]
        minusMinute = str(fulldate-timedelta(minutes=1)).split(" ")[1]
        minusTwoMinutes = str(fulldate-timedelta(minutes=2)).split(" ")[1]
        dateCheck = self.collection.find_one({"date": date})
        if (dateCheck):
            a = self.collection.find_one(
                {"presence.checkIn": {'$in': [hour, minusMinute, minusTwoMinutes]}, "date": date, "presence.perso_id": perso_id})
            print(a)
            if not a:
                self.collection.update_one(
                    {"_id": dateCheck['_id']},
                    {"$push": {"presence": {"perso_id": perso_id, "checkIn": hour,
                                            "checkOut": 'null'}}})

        else:

            self.collection.insert_one(
                {"date": date, "presence": [
                    {"perso_id": perso_id, "checkIn": hour, "checkOut": 'null'}
                ]})

    def checkOutInsert(self, fulldate, perso_id):
        splittedDate = str(fulldate).split(" ")
        date = splittedDate[0]
        hour = splittedDate[1]
        self.collection.update_one(
            {"presence.perso_id": perso_id, "date": date},
            {"$set": {
                'presence.$[xxx].checkOut': hour
            }},
            array_filters=[
                {"xxx.checkOut": 'null'}
            ]
        )

    def checkOutExceptionInsert(self, fulldate, perso_id):
        splittedDate = str(fulldate).split(" ")
        date = splittedDate[0]
        hour = splittedDate[1]
        plusMinute = str(fulldate+timedelta(minutes=1)).split(" ")[1]
        plusTwoMinutes = str(fulldate+timedelta(minutes=2)).split(" ")[1]
        dateCheck = self.collection.find_one({"date": date})
        if (dateCheck):
            a = self.collection.find_one(
                {"presence.checkIn": {'$in': [hour, plusMinute, plusTwoMinutes]}, "date": date, "presence.perso_id": perso_id})
            if not a:
                self.collection.update_one(
                    {"_id": dateCheck['_id']},
                    {"$push": {"presence": {"perso_id": perso_id, "checkIn": 'null',
                               "checkOut": hour}}})
        else:
            self.collection.insert_one(
                {"date": date, "presence": [
                    {"perso_id": perso_id, "checkIn": 'null', "checkOut": hour}
                ]})
