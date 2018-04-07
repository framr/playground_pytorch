"""
This file is a playground used in conjunction with pytest.
Execute
py.test -s  for test run.
Tests not needed currently can be suppressed with X prefix to hide them from pytest
discovery system
"""

import torch.optim as optim
import torch.nn.functional as F
import pytest
from collections import namedtuple
import logging
import sys

from train import train_model
from data import read_dataset, BatchIter
from ffm import FFM
from deepfm import DeepFM


Data = namedtuple("Data", "train_iter test conf")


LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(
    format='[%(asctime)s] %(levelname).1s %(message)s',
    datefmt="%Y.%m.%d %H:%M:%S",
    stream=sys.stdout,
    level=LOGGING_LEVEL
)


@pytest.fixture(scope="module")
def train_data():
    test_file = "./fixtures/net_20180312_201803114_100k_preprocessed.test"
    train_file = "./fixtures/net_20180312_201803114_100k_preprocessed.train"
    fmap_file = "./fixtures/net_20180312_201803114_100k.feature_map"

    num_features = -1
    for _ in open(fmap_file):
        num_features += 1

    conf = {
            "feature_cols": "PageID OrderID".split(),
            "target_col": "IsClick",
            "num_fields": 2,
            "num_features": num_features,
            "dim": 10,
            "use_unary": True,
            "num_iter": 5,
            "opt_cls": optim.Adam,
            "opt_kwargs": {"lr": 1e-3},
            "batch_size": 64
            }

    test = read_dataset(test_file, conf["feature_cols"], conf["target_col"])
    train_iter = BatchIter(train_file, conf["feature_cols"], conf["target_col"], batch_size=conf["batch_size"])
    return Data(train_iter, test, conf)

@pytest.fixture(scope="module")
def train_data():
    test_file = "./fixtures/net_20180312_201803114_100k_preprocessed.test"
    train_file = "./fixtures/net_20180312_201803114_100k_preprocessed.train"
    fmap_file = "./fixtures/net_20180312_201803114_100k.feature_map"

    num_features = -1
    for _ in open(fmap_file):
        num_features += 1
    conf = {
            "feature_cols": "PageID OrderID".split(),
            "target_col": "IsClick",
            "num_fields": 2,
            "num_features": num_features,
            "batch_size": 64
    }
    test = read_dataset(test_file, conf["feature_cols"], conf["target_col"])
    train_iter = BatchIter(train_file, conf["feature_cols"], conf["target_col"], batch_size=conf["batch_size"])
    return Data(train_iter, test, conf)


#@pytest.mark.parametrize("data", [train_100k])
def Xtest_train_ffm(train_data):
    conf = {
            "dim": 10,
            "use_unary": True,
            "num_iter": 5,
            "opt_cls": optim.Adam,
            "opt_kwargs": {"lr": 1e-3},
    }
    conf.update(train_data.conf)
    model = FFM(**conf)
    train_model(model, train_data.train_iter, train_data.test, conf)


def test_train_deepfm(train_data):
    conf = {
            "dim": 10,
            "num_iter": 5,
            "opt_cls": optim.Adam,
            "opt_kwargs": {"lr": 1e-3},
            "deep_layers": [32, 32],
            "deep_activation": F.relu

    }
    conf.update(train_data.conf)

    model = DeepFM(**conf)
    train_model(model, train_data.train_iter, train_data.test, conf)
