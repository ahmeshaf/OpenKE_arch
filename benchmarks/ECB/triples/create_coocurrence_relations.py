from collections import defaultdict



with open("named_ent_source_map.csv") as nf:
    lines = [line.strip().split(',') for line in nf.readlines()]

source_ent_arr = defaultdict(list)

for line in lines:
    source_ent_arr[line[1]].append(line[0])

nary_relations = []
n = 3


def get_nary_relation(ent_array, n=3):
    relations = []
    for i in range(2, n+1):
        rel_name = str(i)+"_ary_cooc"
        for j in range(0, len(ent_array), i):
            nary_array = ent_array[j:j+i]
            for ent1 in nary_array:
                for ent2 in nary_array:
                    if ent1 != ent2:
                        relations.append(",".join([ent1, rel_name, ent2]))
    return relations

with open('ent_cooc_relations.csv', 'w') as ef:
    for ent_array in source_ent_arr.values():
        n_relations = get_nary_relation(ent_array, n)
        if len(n_relations) == 0:
            pass
        elif len(n_relations) == 1:
            ef.write(n_relations[0] + "\n")
        else:
            ef.write("\n".join(n_relations))
            ef.write("\n")
