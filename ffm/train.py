#!/usr/bin/env python3.6
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import logging
from logging import info

from ffm import FFM
from data import batch_iter, read_dataset


LOGGING_LEVEL = logging.DEBUG
LongTensor = torch.LongTensor


def optimizer_factory(conf):
    opt = conf["opt_cls"](**conf["opt_kwargs"])
    return opt


def train_ffm(train, test, conf):
    model = FFM(**conf)
    loss_func = nn.NLLLoss()
    optimizer = optimizer_factory(conf)
    # Loss on test before learning
    test_targets = autograd.Variable(LongTensor(test.y))
    test_features = autograd.Variable(LongTensor(test.X))
    test_logprob = model.forward(test_features)
    test_loss = loss_func(test_logprob, test_targets)
    info("it={it}, test loss={loss}".format(it=-1, loss=float(test_loss)))

    iter_loss = []
    for it in range(conf["num_iter"]):
        data_iter = batch_iter(train, batch_size=conf["batch_size"])
        iter_loss.append(0)
        for batch in data_iter:
            targets = autograd.Variable(LongTensor(batch.y))
            features = autograd.Variable(LongTensor(batch.X))
            model.zero_grad()
            logprob = model.forward(features)
            loss = loss_func(logprob, targets)
            loss.backward()
            optimizer.step()

            iter_loss[-1] += loss.data

        model.zero_grad()
        test_logprob = model.forward(test_features)
        test_loss = loss_func(test_logprob, test_targets)
        info("it={it}, train loss={loss}, test_loss={test}".format(it=it, loss=float(iter_loss[-1]),
                                                                    test=float(test_loss)))


if __name__ == "__main__":
    logging.basicConfig(
        format='[%(asctime)s] %(levelname).1s %(message)s',
        datefmt="%Y.%m.%d %H:%M:%S",
        stream=sys.stdout,
        level=LOGGING_LEVEL
        )

    conf = {"num_fields": 2,
            "dim": 100,
            "num_iter": 1,
            "optimizer": optim.Adam(model.parameters(), lr=conf["lr"])


            #"feat_conf": {
            #    "features": [("PageID", int), ("OrderID", int)],
            #    "target": ("IsClick", int)
            #}
    }

    filename =
    train = read_dataset()

    train_ffm(conf)