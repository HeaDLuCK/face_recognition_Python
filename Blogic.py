from pymongo import MongoClient


class Blogic:
    def __init__(self):
        self.__database = MongoClient(
            "mongodb+srv://zaka:1234@cluster0.uyde97r.mongodb.net/?retryWrites=true&w=majority")
        self.collection = self._Blogic__database.detected_faces.appearance



         

    def checkInInsert(self, fulldate, perso_id):
        date,time = str(fulldate).split(" ")
        hour,minute = map(int,time.split(":"))
        tolal_minute=hour*60+minute
        minusMinute = tolal_minute-1
        minusTwoMinutes = tolal_minute-2
        dateCheck = self.collection.find_one({"date": date})
        if (dateCheck):
            a = self.collection.find_one(
                {"presence.checkIn": {'$in': [tolal_minute, minusMinute, minusTwoMinutes]}, "date": date, "presence.perso_id": perso_id})
            if not a:
                self.collection.update_one(
                    {"_id": dateCheck['_id']},
                    {"$push": {"presence": {"perso_id": perso_id, "checkIn": tolal_minute}}})

        else:

            self.collection.insert_one(
                {"date": date, "presence": [
                    {"perso_id": perso_id, "checkIn": tolal_minute}
                ]})

    def checkOutInsert(self, fulldate, perso_id):
        date,time = str(fulldate).split(" ")
        hour,minute = map(int,time.split(":"))
        tolal_minute=hour*60+minute
        self.collection.update_one(
            {"date": date},
            {"$set": {
                'presence.$[xxx].checkOut': tolal_minute
            }},
            array_filters=[
                {"xxx.perso_id": perso_id, "xxx.checkOut": {"$exists": False}}
            ]
        )

    def checkOutExceptionInsert(self, fulldate, perso_id):
        date,time = str(fulldate).split(" ")
        hour,minute = map(int,time.split(":"))
        tolal_minute=hour*60+minute
        minusMinute = tolal_minute-1
        minusTwoMinute = tolal_minute-2
        dateCheck = self.collection.find_one({"date": date})
        if (dateCheck):
            a = self.collection.find_one(
                {"presence.checkIn": {'$in': [tolal_minute, minusMinute, minusTwoMinute]}, "date": date, "presence.perso_id": perso_id})
            if not a:
                self.collection.update_one(
                    {"_id": dateCheck['_id']},
                    {"$push": {"presence": {"perso_id": perso_id,
                               "checkOut": tolal_minute}}})
        else:
            self.collection.insert_one(
                {"date": date, "presence": [
                    {"perso_id": perso_id, "checkOut": tolal_minute}
                ]})
