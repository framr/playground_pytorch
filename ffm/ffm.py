"""
Simple ffm
"""
import torch
import torch.nn as nn
import math


def glorot(m):
    size = m.weight.size()  # returns a tuple
    fan_out = size[0]  # number of rows
    fan_in = size[1]
    if isinstance(m, nn.Linear):
        scale = math.sqrt(2.0 / (fan_in + fan_out))
    elif isinstance(m, nn.Embedding):
        scale = math.sqrt(2.0 / (1.0 + fan_in * fan_out))
    else:
        raise NotImplementedError
    m.weight.data.uniform_(-scale, scale)


class FFM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FFM, self).__init__()
        self.num_features = kwargs["num_features"]
        self.dim = kwargs["dim"]
        self.num_fields = kwargs["num_fields"]
        self.use_unary = kwargs["use_unary"]

        # create parameters
        self.embeddings = nn.Embedding(self.num_features, self.dim)
        out_dim = self.dim
        if self.use_unary:
            self.unary = nn.Embedding(self.num_features, 1)
            out_dim += self.num_fields
        self.projection = nn.Linear(out_dim, 2)
        # initialize parameters
        glorot(self.embeddings)
        glorot(self.projection)
        if self.use_unary:
            glorot(self.unary)

    def forward(self, X):
        """
        :param self:
        :param X: B (batch size) x F (number of features)
        :return:
        """
        # quadratic cross embeddings: (a+b+c)**2 - a**2 - b**2 - c**2 = 2 * (ab + bc + ac)
        embeddings = self.embeddings(X)  # B x F x D
        embeddings_sum = embeddings.sum(dim=1)  # B x D
        sum_squares = torch.mul(embeddings, embeddings).sum(dim=1)  # B x D
        quadratic = 0.5 * (torch.mul(embeddings_sum, embeddings_sum) - sum_squares)
        if self.use_unary:
            unary = self.unary(X)  # B x F x 1
            unary = unary.squeeze(dim=2)  # B x F
            out = torch.cat((quadratic, unary), dim=1)  # B x (F + D)
        else:
            out = quadratic
        p = self.projection(out)
        logsoftmax = nn.LogSoftmax(dim=1)
        return logsoftmax(p)