from csvutil import csvreader
from collections import namedtuple


Dataset = namedtuple("Dataset", "X y")

def calc_features(example, features, feat_conf):
    return [feat_conf.get(f, int)(example[f]) for f in features]


def read_dataset(filename, feature_cols, target_col, feat_conf=None, batch_size=64, sep="\t", skip_lines=1):
    feat_conf = feat_conf or {}
    X = []
    y = []
    for example in csvreader(filename, sep=sep, skip_lines=skip_lines):
        X.append(calc_features(example, feat_conf))
        y.append(example[target_col])
    return Dataset(X, y)


def batch_iter(filename, feature_cols, target_col, feat_conf=None, batch_size=64, sep="\t", skip_lines=1):
    """
    :param filename:
    :param batch_size:
    :param sep:
    :param skip_lines:
    :return:
    """
    feat_conf = feat_conf or {}
    Xb = []
    yb = []
    for example in csvreader(filename, sep=sep, skip_lines=skip_lines):
        if len(Xb) >= batch_size:
            yield Dataset(Xb, yb)
            Xb = []
            yb = []
        Xb.append(calc_features(example, feature_cols, feat_conf))
        yb.append(example[target_col])
    if Xb:
        yield Dataset(Xb, yb)
