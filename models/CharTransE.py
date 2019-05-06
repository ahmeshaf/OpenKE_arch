import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np



from scipy.sparse import coo_matrix
import os
from scipy import sparse
import re
from .Model import Model
from torch.nn.modules.sparse import Embedding

def sparse_tensor(vector):
    coo = coo_matrix(vector)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def partition(some_list, length=3):
    for i in range(0, len(some_list)):
        yield some_list[i:i + length]

def transform_into_substring(some_string, sub_len = 3):
    if some_string.startswith('kg:') or some_string.startswith('tp:'):
        return some_string
    all_subs = []
    clean_string = re.sub('[\.\,\:]', '', some_string)
    clean_string = re.sub('\s', '_', clean_string)
    words = [w.strip() for w in clean_string.split('_') if w.strip() != '']
    all_subs.extend(words)
    for word in words:
        all_subs.extend(partition(word, sub_len))
    if len(words) > 1:
        bridge_subs = ['_'.join([first[-1],second[0]]) for first,second in zip(words[:-1], words[1:])]
        all_subs.extend(bridge_subs)
    return ' '.join(all_subs)

class CharTransE(Model):
    def __init__(self, config):
        super(CharTransE, self).__init__(config)
        sparse_path = os.path.dirname(__file__) + '/ent_tfidf.pkl.npz'
        self.tf_idf_mat = sparse.load_npz(sparse_path)

        ent_shape = self.tf_idf_mat.shape[1]

        # self.tfidf_embedding = Embedding(tf_idf_mat.shape[0],ent_shape, sparse=True)
        # self.tfidf_embedding.weight.requires_grad = False
        self.tfidf_embedding = nn.Embedding.from_pretrained(sparse_tensor(self.tf_idf_mat), freeze=True, sparse=True)
        # self.tfidf_embedding = Embedding(self.tf_idf_mat.shape[0], ent_shape, sparse=True)
        # self.tfidf_embedding.load_state_dict({'weight':self.tf_idf_mat})
        # self.tfidf_embedding.weight.requires_grad=False
        self.ent_char_weights = nn.Linear(ent_shape, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_char_weights.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cpu())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        # h_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_h.numpy(), :])
        h_sparse_x = self.tfidf_embedding(self.batch_h)
        # t_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_t.numpy(), :])
        t_sparse_x = self.tfidf_embedding(self.batch_t)
        h = self.ent_char_weights(h_sparse_x)
        t = self.ent_char_weights(t_sparse_x)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)


    def predict(self):
        h_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_h.numpy(), :])
        t_sparse_x = sparse_tensor(self.tf_idf_mat[self.batch_t.numpy(), :])
        h = self.ent_char_weights(h_sparse_x)
        t = self.ent_char_weights(t_sparse_x)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()
