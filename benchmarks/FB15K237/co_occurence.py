from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
coll = client['freebase']['fb15k237']

with open('entity2id.txt') as ef:
    lines = ef.readlines()[1:]


