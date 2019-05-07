from pymongo import MongoClient
import re

client = MongoClient("mongodb://localhost:27017/")
coll = client['freebase']['fb15k237']

def name_map():
    names = sorted(coll.distinct('name'))

    types = sorted([type for type in coll.distinct('@type')])
    all_names = types + names
    with open('name2id.txt', 'w') as nf:
        nf.write(str(len(all_names)) + "\n")
        nf.write("\n".join(['\t'.join([re.sub('\s', '_',name), str(i)]) for i, name in enumerate(all_names)]))
    # print(names[65])

name_map()
