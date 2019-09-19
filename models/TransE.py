import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from torch.nn.functional import cosine_similarity
class TransE(Model):
	def __init__(self, config):
		super(TransE, self).__init__(config)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
		self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)

		# Adding Weights to learn f(h,r) ~ t
		self.non_linear_weights = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size, bias=False).cpu()

		self.criterion = nn.MarginRankingLoss(self.config.margin, False)
		self.init_weights()
		
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)

		nn.init.xavier_uniform(self.non_linear_weights.weight.data)

	def _calc(self, h, t, r):
		return torch.norm(h + r - t, self.config.p_norm, -1)

	def _calc3(self, h, t, r):
		return 1 - cosine_similarity(h+r, t)

	# calc norm of f(h,r) - t
	def _calc2(self, h, t, r):

		h_r = torch.cat((h,r), 1)
		f_h_r = self.non_linear_weights(h_r)
		return torch.norm(f_h_r - t, self.config.p_norm, -1)
	
	def loss(self, p_score, n_score):
		y = Variable(torch.Tensor([-1]).cpu())
		return self.criterion(p_score, n_score, y)

	def forward(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		# score = self._calc2(h ,t, r)
		score = self._calc(h ,t, r)
		p_score = self.get_positive_score(score)
		n_score = self.get_negative_score(score)
		return self.loss(p_score, n_score)	
	def predict(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		# score = self._calc2(h, t, r)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()
