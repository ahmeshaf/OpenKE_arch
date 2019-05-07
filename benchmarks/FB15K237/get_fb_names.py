"""Example of Python client calling Knowledge Graph Search API."""
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from pymongo import MongoClient, ASCENDING
# query = 'Taylor Swift'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
api_key = "[API_KEY]"

db_name = 'freebase'
collection_name = 'fb15k237'
MONGO_DB_URL = "mongodb://localhost:27017/" # currently just using a local db

def get_names():
    client = MongoClient(MONGO_DB_URL)
    coll = client[db_name][collection_name]
    coll.create_index([("@id", ASCENDING)])
    with open('entity2id.txt') as ef:
        entities = [line.split('\t')[0] for line in ef.readlines()[1:-1]]
        step = 150

        index = 0

        while index <= len(entities):
            ids = [{'ids': ent} for ent in entities][index:index+step]

            params = {
                'limit':len(ids),
                # 'indent':1,
                'key': api_key
            }

            ids_string = "&".join([urlencode(i) for i in ids])
            if len(ids) == 1:
                ids_string = urlencode(ids[0])
            elif len(ids) == 0:
                index += step
                continue

            url = service_url + '?' + urlencode(params) + '&' + ids_string
            successful = False
            while not successful:
                try:
                    response = json.loads(urlopen(url).read())
                    successful = True
                except:
                    print("Error on index: {}".format(index))
            with open('json/{}.json'.format(index), 'w') as jf:
                jf.write(str([res['result'] for res in response['itemListElement']]))
            index += step
        # url = 'https://kgsearch.googleapis.com/v1/entities:search?ids=%2Fm%2F0gyh&ids=%2Fm%2F0gyh&limit=2&key=AIzaSyBdCB1Y-RZ2C0eubbGRPtTYsZeXFuaBUb8'

        # print(response)

get_names()

# print(urlencode({
#             'limit':1,
#             # 'indent':False,
#             'key': api_key
#         }))
# params = {
#     'query': query,
#     'limit': 10,
#     'indent': True,
#     'key': api_key,
# }
# url = service_url + '?' + urllib.urlencode(params)
# response = json.loads(urllib.urlopen(url).read())
# for element in response['itemListElement']:
#   print element['result']['name'] + ' (' + str(element['resultScore']) + ')'
