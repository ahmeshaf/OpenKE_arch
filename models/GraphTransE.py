import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from models import Model
from collections import defaultdict
import pandas as pd
from config import Config
from math import isnan

from scipy.sparse import coo_matrix
import os
from scipy import sparse
from models.Model import Model
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from models.CharTransE import transform_into_substring, sparse_tensor


def initialize_tfidf(config):
    in_path = config.in_path
    ent_eve_name_path = in_path + "/triples/ent_eve_name_map.csv"
    ent_eve_type_path = in_path + "/triples/ent_eve_type_map.csv"
    eve_map_path = in_path + "/triples/eve_map.csv"
    entity2id_file = in_path + '/entity2id.txt'
    name_type_dict = defaultdict(list)
    with open(eve_map_path) as ef:
        events = set([line.split(',')[0] for line in ef.readlines()])
    with open(entity2id_file) as ef:
        entities = [line.split('\t')[0] for line in ef.readlines()[1:]]
    name_df = pd.read_csv(ent_eve_name_path, header=None)
    all_names_and_types = set()
    for row in name_df.itertuples(index=0):
        if row[0] in events:
            name_type_dict[row[0]].append(row[0])
            all_names_and_types.add(row[0])
        if type(row[1]) == str:
            name_type_dict[row[0]].append(row[1])
            all_names_and_types.add(row[1])

    type_df = pd.read_csv(ent_eve_type_path, header=None)
    for row in type_df.itertuples(index=0):
        if type(row[1]) == str:
            name_type_dict[row[0]].append(row[1])
            all_names_and_types.add(row[1])



    docs = [transform_into_substring(nt) for nt in all_names_and_types]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs)
    ent_docs = []
    for ent in entities:
        doc = ' '.join([transform_into_substring(val) for val in name_type_dict[ent]])
        ent_docs.append(doc)
    return vectorizer.transform(ent_docs)




class GraphTransE(Model):
    def __init__(self, config):
        super(GraphTransE, self).__init__(config)

        if config.tfidf == True:
            self.tf_idf_mat = initialize_tfidf(config)
            sparse.save_npz(config.in_path + '/ent_tfidf.pkl.npz', self.tf_idf_mat)
        else:
            sparse_path = config.in_path + 'ent_tfidf.pkl.npz'
            self.tf_idf_mat = sparse.load_npz(sparse_path)

        # # Read the enitityToId file know the number of entities.
        # entity2id_file = config.in_path + '/entity2id.txt'
        # with open(entity2id_file) as ef:
        #     num_ents = int(ef.readline().strip())
        #
        # # Read the entity2type file to read the ontological type of the entities
        # entityid2type_file = config.in_path + '/entityid2type.txt'
        # with open(entityid2type_file) as ef:
        #     type_lines = ef.readlines()[1:] # First line stores the num of entities. skip it
        #
        # # create entityid2types dictionary
        # entityid2types_dict = {} # Initialize the types dict
        # for line in type_lines:
        #     line_split = line.split("\t")
        #     id = int(line_split[0])
        #     types = line_split[1:]
        #     entityid2types_dict[id] = types

        ent_shape = self.tf_idf_mat.shape[1]


        # self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size).cpu()
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.ent_char_weights = nn.Linear(ent_shape, self.config.hidden_size, bias=True).cpu()

        self.init_weights()


    def init_weights(self):
        # nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_char_weights.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cpu())
        return self.criterion(p_score, n_score, y)

    def forward(self):

        h_tf = sparse_tensor(self.tf_idf_mat[self.batch_h.cpu().data.numpy(), :])
        # h_tf = h_tf.mul(self.attention)
        h_sparse_x = h_tf.cpu()
        # h_sparse_x = torch.mul(h_sparse_x, self.attention)
        # h_sparse_x = self.tfidf_embedding(self.batch_h)
        t_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_t.cpu().data.numpy(), :]).cpu()
        # t_sparse_x = self.tfidf_embedding(self.batch_t)
        h = self.ent_char_weights(h_sparse_x)
        t = self.ent_char_weights(t_sparse_x)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)


    def predict(self):
        h_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_h.cpu().data.numpy(), :]).cpu()
        t_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_t.cpu().data.numpy(), :]).cpu()
        h = self.ent_char_weights(h_sparse_x)
        t = self.ent_char_weights(t_sparse_x)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()

if __name__=='__main__':
    config = Config()
    config.in_path = '../benchmarks/ECB'
    initialize_tfidf(config)