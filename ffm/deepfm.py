import math
import torch
import torch.nn as nn


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


class DeepFM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FFM, self).__init__()
        self.num_features = kwargs["num_features"]
        self.dim = kwargs["dim"]
        self.num_fields = kwargs["num_fields"]
        self.deep_layers = kwargs["deep_layers"]
        self.num_deep_layers = len(self.deep_layers)

        # create parameters
        # embeddings
        self.embeddings = nn.Embedding(self.num_features, self.dim)
        self.out_dim = self.dim

        # unary weights
        self.unary = nn.Embedding(self.num_features, 1)
        self.out_dim += self.num_fields

        # deep weights
        if self.deep_layers:
            self.deep_0 = nn.Linear(self.num_fields * self.dim, self.deep_layers[0])
        for i in range(1, self.num_deep_layers):
            setattr(self, "deep_%d" % i, nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            self.out_dim += self.deep_layers[-1]

        self.projection = nn.Linear(self.out_dim, 1)
        # initialize parameters
        glorot(self.embeddings)
        glorot(self.projection)
        glorot(self.unary)
        for i in range(self.num_deep_layers):
            glorot(getattr(self, "deep_%i"))

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
        unary = self.unary(X)  # B x F x 1
        unary = unary.squeeze(dim=2)  # B x F
        out = torch.cat((quadratic, unary), dim=1)  # B x (F + D)

        y_deep = embeddings.view(-1, self.num_fields * self.dim)
        for i in range(self.num_deep_layers):
            y_deep = getattr(self, "deep_%i")()

        logsigmoid = nn.LogSigmoid()
        return logsigmoid(out)
