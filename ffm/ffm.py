"""
Simple ffm
"""
import torch
import torch.nn as nn
import math


class FFM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FFM, self).__init__()

        self.num_features = kwargs["num_features"]
        self.dim = kwargs["dim"]
        self.embeddings = nn.Embedding(self.num_features, self.dim)
        self.unary = nn.Embedding(self.num_features, 1)
        #self.logsigmoid = nn.LogSigmoid()

        # initialize weights
        glorot = math.sqrt(2.0 / (self.num_features * self.dim + 1.0))
        self.embeddings.weight.data.uniform_(-glorot, glorot)
        glorot = math.sqrt(2.0 / (self.num_features + 1.0))
        self.unary.weight.data.uniform_(-glorot, glorot)

    def forward(self, X):
        """
        :param self:
        :param X: B (batch size) x F (number of features)
        :return:
        """
        embeddings = self.embeddings(X)  # B x F x D
        embeddings_sum = torch.sum(dim=1)  # B x 1 x D
        sum_squares = torch.mul(embeddings, embeddings).sum(dim=1)  # B x 1 x D
        quadratic = torch.mul(embeddings_sum, embeddings_sum) - sum_squares
        unary = self.unary(X)  # B x F x 1
        logsigmoid = nn.LogSigmoid(quadratic + unary)
        return logsigmoid
