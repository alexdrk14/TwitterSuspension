""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
MongoDB connection class
####################################################################################################################"""

import config as cnf
from pymongo import MongoClient

class MongoDB:
    def __init__(self):
        self.ip = cnf.ip
        self.port = cnf.port
        self.DBname = cnf.DBNAME
       
    def connect(self):
        client = MongoClient(self.ip, port = self.port)
        db = client[self.DBname]
        return client, db
