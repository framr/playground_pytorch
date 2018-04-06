from csvutil import csvreader
from collections import namedtuple


Dataset = namedtuple("Dataset", "X y")


def calc_features(example, features, feat_conf):
    return [feat_conf.get(f, int)(example[f]) for f in features]


def read_dataset(filename, feature_cols, target_col, feat_conf=None, sep="\t", skip_lines=1):
    feat_conf = feat_conf or {}
    X = []
    y = []
    for example in csvreader(filename, sep=sep, skip_lines=skip_lines):
        X.append(calc_features(example, feature_cols, feat_conf))
        y.append(int(example[target_col]))
    return Dataset(X, y)


class BatchIter:
    def __init__(self, filename, feature_cols, target_col, feat_conf=None, batch_size=64, sep="\t", skip_lines=1):
        self.filename = filename
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.feat_conf = feat_conf or {}
        self.batch_size = batch_size
        self.sep = sep
        self.skip_lines = skip_lines
        self.stream = csvreader(filename, sep=sep, skip_lines=skip_lines)
        self.Xb = None
        self.yb = None

    def __iter__(self):
        return self

    def __next__(self):
        self.Xb = []
        self.yb = []
        while len(self.Xb) < self.batch_size:
            try:
                example = next(self.stream)
            except StopIteration:
                if self.Xb:
                    return Dataset(self.Xb, self.yb)
                else:
                    raise
            self.Xb.append(calc_features(example, self.feature_cols, self.feat_conf))
            self.yb.append(int(example[self.target_col]))
        return Dataset(self.Xb, self.yb)

    def reset(self):
        self.stream = csvreader(self.filename, sep=self.sep, skip_lines=self.skip_lines)


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
