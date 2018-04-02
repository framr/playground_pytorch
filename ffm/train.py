#!/usr/bin/env python3.6
import torch.nn as nn

from ffm import FFM
from data import batch_iter


def train(conf):
    model = FFM(**conf)
    loss = nn.NLLLoss()
    num_iter = conf["num_iter"]

    biter = batch_iter(conf["feat_conf"], batch_size=conf["batch_size"], skip_lines=1)
    for it in range(num_iter):
        print("Iteration {iter}".formar(iter=it))
        for Xb, yb in biter:

        biter = batch_iter(conf["feat_conf"], batch_size=conf["batch_size"], skip_lines=1)


if __name__ == "__main__":
    conf = {"num_features": 100,
            "dim": 100,
            "num_iter": 1,
            "feat_conf": {
                "features": [("PageID", int), ("OrderID", int)],
                "target": ("IsClick", int)
            }
    }
    train(conf)



