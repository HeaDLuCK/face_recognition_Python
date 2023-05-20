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


    def checkInInsert(self, fulldate, perso_id):
        splittedDate = str(fulldate).split(" ")
        date = splittedDate[0]
        hour = splittedDate[1]
        minute = int(hour.split(':')[0])*60+int(hour.split(':')[1])
        minusMinute = minute-1
        minusTwoMinutes = minute-2
        dateCheck = self.collection.find_one({"date": date})
        if (dateCheck):
            a = self.collection.find_one(
                {"presence.checkIn": {'$in': [minute, minusMinute, minusTwoMinutes]}, "date": date, "presence.perso_id": perso_id})
            if not a:
                self.collection.update_one(
                    {"_id": dateCheck['_id']},
                    {"$push": {"presence": {"perso_id": perso_id, "checkIn": minute}}})

        else:

            self.collection.insert_one(
                {"date": date, "presence": [
                    {"perso_id": perso_id, "checkIn": minute}
                ]})

    def checkOutInsert(self, fulldate, perso_id):
        splittedDate = str(fulldate).split(" ")
        date = splittedDate[0]
        hour = splittedDate[1]
        minute = int(hour.split(':')[0])*60+int(hour.split(':')[1])
        self.collection.update_one(
            {"date": date},
            {"$set": {
                'presence.$[xxx].checkOut': minute
            }},
            array_filters=[
                {"xxx.perso_id": perso_id, "xxx.checkOut": {"$exists": False}}
            ]
        )

    def checkOutExceptionInsert(self, fulldate, perso_id):
        splittedDate = str(fulldate).split(" ")
        date = splittedDate[0]
        hour = splittedDate[1]
        minute = int(hour.split(':')[0])*60+int(hour.split(':')[1])
        minusMinute = minute-1
        minusTwoMinute = minute-2
        dateCheck = self.collection.find_one({"date": date})
        if (dateCheck):
            a = self.collection.find_one(
                {"presence.checkIn": {'$in': [minute, minusMinute, minusTwoMinute]}, "date": date, "presence.perso_id": perso_id})
            if not a:
                self.collection.update_one(
                    {"_id": dateCheck['_id']},
                    {"$push": {"presence": {"perso_id": perso_id,
                               "checkOut": minute}}})
        else:
            self.collection.insert_one(
                {"date": date, "presence": [
                    {"perso_id": perso_id, "checkOut": minute}
                ]})
