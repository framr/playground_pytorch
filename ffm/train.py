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


def optimizer_factory(model, conf):
    opt = conf["opt_cls"](model.parameters(), **conf["opt_kwargs"])
    return opt


def train_ffm(train_iter, test, conf):
    model = FFM(**conf)
    loss_func = nn.NLLLoss()
    optimizer = optimizer_factory(model, conf)

    # Loss on test before learning
    test_targets = autograd.Variable(LongTensor(test.y))
    test_features = autograd.Variable(LongTensor(test.X))
    test_logprob = model.forward(test_features)
    test_loss = loss_func(test_logprob, test_targets)
    info("it={it}, test loss={loss}".format(it=-1, loss=float(test_loss)))

    iter_loss = []
    for it in range(conf["num_iter"]):
        train_iter.reset()
        iter_loss.append(0)
        for batch in train_iter:
            targets = autograd.Variable(LongTensor(batch.y))
            features = autograd.Variable(LongTensor(batch.X))
            model.zero_grad()
            logprob = model.forward(features)
            loss = loss_func(logprob, targets)
            loss.backward()
            optimizer.step()
            iter_loss[-1] += loss.data

        test_logprob = model.forward(test_features)
        test_loss = loss_func(test_logprob, test_targets)
        info("it={it}, train loss={loss}, test_loss={test}".format(it=it, loss=float(iter_loss[-1]),
                                                                       test=float(test_loss)))
        #print(float(test_loss))
