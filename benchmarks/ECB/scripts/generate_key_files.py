from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict, OrderedDict

red_win_doc_cluster_file = "event_win_ecb.txt"
red_cross_doc_cluster_file = "event_clusters_ecb.txt"

# ta2_win_doc_cluster_file = "bbn/just_win_doc.txt"
ta2_win_doc_cluster_file = "eve_5/event_win_clus.txt"
# ta2_cross_doc_cluster_file = "bbn/cluster_just_map.txt"
# ta2_cross_doc_cluster_file = "isi_2018/cluster_just_map_rel_vec.txt"
# ta2_cross_doc_cluster_file = "isi_2018/cluster_just_map_ref.txt"
ta2_cross_doc_cluster_file = "eve_5/event_clusters.txt"



def get_clus_dict(cluster_file):
    """
    Read file where each line is just_id , clus_id
    :param cluster_file:
    :return:
    """
    with open(cluster_file) as cf:
        lines = [line.strip().split(',') for line in cf.readlines()]
        clus_dict = {line[0]:line[1].strip('(').strip(')') for line in lines}
    return clus_dict

def get_clus_array(clus_dict):
    """
    Convert the cluster dictionary into array of arrays
    :param clus_dict: dict
    :return: array
    """
    clus_array_dict = defaultdict(list)
    for key, value in clus_dict.items():
        clus_array_dict[value].append(key)
    return clus_array_dict.values()



def get_just2ent_map(red_win_clus_dict, ta2_win_clus_dict):
    """
    With the within doc clusters of justifications merge the clusters coming from ta2 ttl files and red files
    :param red_win_doc_clus:
    :param ta2_win_doc_clus:
    :return: entity names, just2ent map
    """
    just_set = sorted(list(set(list(red_win_clus_dict.keys()) + list(ta2_win_clus_dict.keys()))))
    n = len(just_set)

    # entity_names = [""]

    just2id = {just:i for i, just in enumerate(just_set)}
    red_clus_array = get_clus_array(red_win_clus_dict)
    ta2_clus_array = get_clus_array(ta2_win_clus_dict)
    red_adj_mat = lil_matrix((n, n))
    ta2_adj_mat = lil_matrix((n, n))

    for cluster in red_clus_array:
        for ent1 in cluster:
            for ent2 in cluster:
                red_adj_mat[just2id[ent1], just2id[ent2]] = 1

    for cluster in ta2_clus_array:
        for ent1 in cluster:
            for ent2 in cluster:
                ta2_adj_mat[just2id[ent1], just2id[ent2]] = 1

    merged_adj_mat = (red_adj_mat + ta2_adj_mat).astype('bool')
    _, labels = connected_components(csgraph=merged_adj_mat, directed=False, return_labels=True)

    entity_names = ["entity_%d"%i for i in range(max(labels) + 1)]

    return entity_names, {just:"entity_%d"%lab for just, lab in zip(just_set, labels)}

def get_entity_map(entity_names, just2ent, clus_dict):
    """
    return an ordered dict with ent to clus map
    :param entity_names:
    :param just2ent:
    :param clus_dict: just to clus
    :return:
    """
    ent_clus_dict = OrderedDict()
    for ent in entity_names:
        ent_clus_dict[ent] = "-"

    for just, clus in clus_dict.items():
        if just in just2ent:
            ent = just2ent[just]
            ent_clus_dict[ent] = clus

    return ent_clus_dict

def create_key_file(ent_clus_dict, file_name, key_name):
    # key_name = "RED"
    values = ent_clus_dict.values()
    value2_ind = {}
    clus_num = 0
    for val in values:
        if val not in value2_ind:
            value2_ind[val] = clus_num
            clus_num+=1
    value2_ind['-'] = '-'
    with open(file_name, 'w') as kf:
        kf.write("#begin document (%s);\n" % key_name)
        for i, item in enumerate(ent_clus_dict.items()):
            kf.write("%s\t0\t%d\t%s\t(%s)\n" % (key_name, i, item[0], value2_ind[item[1]]))
        kf.write("#end document\n")


if __name__=="__main__":
    red_win_doc_dict = get_clus_dict(red_win_doc_cluster_file)
    red_cross_doc_dict = get_clus_dict(red_cross_doc_cluster_file)

    ta2_win_doc_dict = get_clus_dict(ta2_win_doc_cluster_file)
    ta2_cross_doc_dict = get_clus_dict(ta2_cross_doc_cluster_file)

    ent_names, just2ent = get_just2ent_map(red_win_doc_dict, ta2_win_doc_dict)

    red_corss_ent_map = get_entity_map(ent_names, just2ent, red_cross_doc_dict)
    ta2_cross_ent_map = get_entity_map(ent_names, just2ent, ta2_cross_doc_dict)

    key_name = "ECB_TEST"
    create_key_file(red_corss_ent_map, 'ecb_cross_key.txt', key_name)
    # create_key_file(red_corss_ent_map, 'red_cross_key_ref.txt', key_name)
    # create_key_file(ta2_cross_ent_map, 'ta2_cross_key_vec.txt', key_name)
    create_key_file(ta2_cross_ent_map, 'ta2_cross_key.txt', key_name)