import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch.autograd import Variable
import numpy as np



from scipy.sparse import coo_matrix
import os
from scipy import sparse
from models.Model import Model
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

    if (some_string.startswith('type:')):
        types = some_string.split('.')
        return ' '.join([some_string]+ types[:1] + ['type:' + typ for typ in types[1:]])
    elif 'ecb_' in some_string or some_string.startswith("http"):
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
        if config.tfidf == True:
            ent_path = config.in_path + '/entity2id.txt'
            with open(ent_path) as ff:
                lines = ff.readlines()[1:]
                entities = [line.split('\t')[0] for line in lines]
                docs = [transform_into_substring(ent) for ent in entities]
                vectorizer = TfidfVectorizer()
                self.tf_idf_mat = vectorizer.fit_transform(docs)
                sparse.save_npz(config.in_path + '/ent_tfidf.pkl.npz', self.tf_idf_mat)
                # save_tfidf(docs)
        else:
            sparse_path = config.in_path + 'ent_tfidf.pkl.npz'
            self.tf_idf_mat = sparse.load_npz(sparse_path)

        # sparse_path = os.path.dirname(__file__) + '/ent_tf.pkl.npz'
        # sparse_path = config.in_path + 'ent_tfidf.pkl.npz'
        # self.tf_idf_mat = sparse.load_npz(sparse_path)

        ent_shape = self.tf_idf_mat.shape[1]

        # self.attention = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(1, ent_shape)))

        # self.tfidf_embedding = Embedding(tf_idf_mat.shape[0],ent_shape, sparse=True)
        # self.tfidf_embedding.weight.requires_grad = False
        # self.tfidf_embedding = nn.Embedding.from_pretrained(self.tf_idf_mat, freeze=True)
        # self.tfidf_embedding = Embedding(self.tf_idf_mat.shape[0], ent_shape, sparse=True)
        # self.tfidf_embedding.load_state_dict({'weight':self.tf_idf_mat})
        # self.tfidf_embedding.weight.requires_grad=False
        self.ent_char_weights = nn.Linear(ent_shape, self.config.hidden_size, bias=True).cpu()
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size).cpu()
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def ent_similarity(self, ent1, ent2):
        embed1 = sparse_tensor(self.tf_idf_mat[ent1, :])

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_char_weights.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

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


def save_tfidf(docs):
    vectorizer = TfidfVectorizer()
    # vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    # pickle.dump(X, open('ent_tfidf.pkl', 'wb'))
    sparse.save_npz("../benchmarks/tim_minikb/ent_tfidf.pkl", X)


if __name__=='__main__':
    file = '../benchmarks/tim_minikb/entity2id.txt'
    with open(file) as ff:
        lines = ff.readlines()[1:]
        entities = [line.split('\t')[0] for line in lines]
        docs = [transform_into_substring(ent) for ent in entities]
        save_tfidf(docs)

    # X = sparse.load_npz('ent_tfidf.pkl.npz')
    X = sparse.load_npz('ent_tf.pkl.npz')
    print(X.shape)
    # x_ten = convert_to_tensor(X[[10, 12], :])
    pass