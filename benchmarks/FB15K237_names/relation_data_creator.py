from pymongo import MongoClient
import re

client = MongoClient("mongodb://localhost:27017/")
coll = client['freebase']['fb15k237']

start_count = 14541

with open('./entity2id_old.txt') as ef:
    all_lines = ef.readlines()[1:]
    lines = [line.split('\t') for line in all_lines]
    ent2id = {"kg:" + line[0]: int(line[1]) for line in lines}
    id2ent = {int(line[1]): "kg:" + line[0] for line in lines}

unique_types = sorted([ 'tp:' + typ for typ in coll.distinct('@type')])
unique_types.remove('tp:Thing')
print(len(unique_types))

for typ in unique_types:
    all_lines.append('\t'.join([typ, str(start_count)])+'\n')
    ent2id[typ] = str(start_count)
    id2ent[start_count] = typ
    start_count+=1

names = sorted([ name for name in coll.distinct('name')])
for name in names:
    all_lines.append('\t'.join([name, str(start_count)])  + '\n')
    ent2id[name] = str(start_count)
    id2ent[start_count] = name
    start_count+=1


with open('./entity2id.txt', 'w') as ef:
    ef.write(str(len(all_lines)) + '\n')
    ef.write(''.join(all_lines))



rel2id = {'sameAs':'237', 'type':'238', 'name':'239'}

all_entries = coll.find({}, {'@id':1, '@type':1, 'name':1, 'description':1})
relations = []
index = 1
for entry in all_entries:
    ent_id = entry['@id']
    ent_types = entry['@type']

    if not ent_id in ent2id:
        continue
    index+=1
    id_ = str(ent2id[ent_id])
    relations.append((id_, id_, rel2id['sameAs']))
    for typ in ent_types:
        if not 'Thing' in typ:
            relations.append((id_, str(ent2id['tp:'+typ]), rel2id['type']))
            pass

    if 'name' in entry:
        ent_name = entry['name']
        name_id = str(ent2id[ent_name])
        relations.append((id_, name_id, rel2id['name']))
        relations.append((name_id, name_id, rel2id['sameAs']))
    pass
print(index)
print(len(relations))
pass
with open('train2id_old.txt') as tf:
    t_lines = tf.readlines()[1:]
for relation in relations:
    t_lines.append(' '.join(relation) + '\n')

with open('train2id.txt', 'w') as tf:
    tf.write(str(len(t_lines)) + '\n')
    tf.write(''.join(t_lines))

