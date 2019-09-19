import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model


class GraphTransE(Model):
    def __init__(self, config):
        super(GraphTransE, self).__init__(config)

        # Read the enitityToId file know the number of entities.
        entity2id_file = config.in_path + '/entity2id.txt'
        with open(entity2id_file) as ef:
            num_ents = int(ef.readline().strip())

        # Read the entity2type file to read the ontological type of the entities
        entityid2type_file = config.in_path + '/entityid2type.txt'
        with open(entityid2type_file) as ef:
            type_lines = ef.readlines()[1:] # First line stores the num of entities. skip it

        # create entityid2types dictionary
        entityid2types_dict = {} # Initialize the types dict
        for line in type_lines:
            line_split = line.split("\t")
            id = int(line_split[0])
            types = line_split[1:]
            entityid2types_dict[id] = types


        # self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)


        ent_shape = 1000
        ent_char_shape = 10000

        self.ent_type_weights = nn.Linear(ent_shape, self.config.hidden_size, bias=True).cpu()
        self.char_weights = nn.Linear(ent_char_shape, self.config.hidden_size, bias=True).cpu()

        self.non_linear_weights = nn.Linear(2*self.config.hidden_size, self.config.hidden_size, bias=False).cpu()

        self.init_weights()


    def init_weights(self):
        # nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cuda())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()
