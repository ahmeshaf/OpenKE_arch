import ast
import os
from pymongo import MongoClient
db_name = 'freebase'
collection_name = 'fb15k237'
MONGO_DB_URL = "mongodb://localhost:27017/" # currently just using a local db

files = os.listdir('json/')
for file in files:
    with open('json/'+file) as jf:
        data = ast.literal_eval(jf.read())
    client = MongoClient(MONGO_DB_URL)
    coll = client[db_name][collection_name]
    coll.insert_many(data)