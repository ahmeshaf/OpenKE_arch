from collections import Counter
from random import shuffle
import pandas
from math import isnan

with open('triples/eve_arg_triples.csv') as rf:
    all_triples = [line.strip().split(',') for line in rf.readlines()]

name_map_df = pandas.read_csv('triples/ent_name_map.csv')
named_ents = set()
for row in name_map_df.itertuples(index=False):
    if type(row[1]) == str:
        named_ents.add(row[0])


# use only named entities
triples = []
ents = []
rels = []

for trip in all_triples:

    triples.append(trip)
    ents.append(trip[0].strip())
    ents.append(trip[2].strip())
    rels.append(trip[1])

# triples = all_triples[:]

shuffle_trips = triples[:]
shuffle(shuffle_trips)
id_ents = sorted([ent for ent in set(ents) ])
# name_ents = sorted([ent for ent in set(ents)])

rels = sorted(set(rels))

# ents = id_ents + name_ents

u_ents = {ent:str(i) for i, ent in enumerate(id_ents)}
u_rels = {rel:str(i) for i, rel in enumerate(rels)}

# u_rels['sameAs'] = str(len(u_rels))

with open('entity2id.txt', 'w') as ef:
    order_ents = sorted(u_ents.items(), key=lambda x: int(x[1]))
    ef.write(str(len(order_ents)) + '\n')
    ef.write('\n'.join(['\t'.join(item) for item in order_ents]))
    ef.write('\n')

with open('relation2id.txt', 'w') as ef:
    order_rels = sorted(u_rels.items(), key=lambda x: int(x[1]))
    ef.write(str(len(order_rels)) + '\n')
    ef.write('\n'.join(['\t'.join(item) for item in order_rels]))
    ef.write('\n')

with open('train2id.txt', 'w') as tf:
    tf.write(str(len(triples)) + '\n')
    for trip in triples:
        tf.write(' '.join([u_ents[trip[0].strip()], u_ents[trip[2].strip()], u_rels[trip[1].strip()]]))
        tf.write('\n')
    # for ent in u_ents.values():
    #     tf.write(' '.join([ent, ent, u_rels['sameAs']]))
    #     tf.write('\n')

with open('valid2id.txt', 'w') as tf:
    valid_trips = shuffle_trips[:300]
    tf.write(str(len(valid_trips)) + '\n')
    for trip in valid_trips:
        tf.write(' '.join([u_ents[trip[0].strip()], u_ents[trip[2].strip()], u_rels[trip[1].strip()]]))
        tf.write('\n')

with open('test2id.txt', 'w') as tf:
    shuffle(shuffle_trips)
    test_trips = shuffle_trips[:300]
    tf.write(str(len(test_trips)) + '\n')
    for trip in test_trips:
        tf.write(' '.join([u_ents[trip[0].strip()], u_ents[trip[2].strip()], u_rels[trip[1].strip()]]))
        tf.write('\n')





