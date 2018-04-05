#!/usr/bin/env python3.6
from collections import defaultdict
from collections import namedtuple
import logging
from logging import info
import sys
import random

from csvutil import csvreader


LOGGING_LEVEL = logging.DEBUG
FeatureStats = namedtuple("FeatureStats", "namespace fid counts")


def calc_feature_stats(filename, features, outfilename, sep="\t", skip_lines=1):
    """
    Calculate feature -> counts statistics.
    :param filename:
    :return:
    """
    fstats = defaultdict(dict)
    for rec in csvreader(filename, sep=sep, skip_lines=skip_lines):
        for ns in features:
            fids = rec[ns].split()
            for fid in fids:
                fstats[ns][fid] = fstats[ns].get(fid, 0) + 1
    with open(outfilename, "w") as fd:
        fd.write("namespace\tfeature\tcounts\n")
        for fname, fids in fstats.items():
            for fid, counts in fids.items():
                fd.write("{fname}\t{feature}\t{counts}\n".format(fname=fname, feature=fid, counts=fstats[fname][fid]))


def gen_feature_stats(filename, sep="\t", skip_lines=1):
    for rec in csvreader(filename, sep=sep, skip_lines=skip_lines):
        yield FeatureStats(rec["namespace"], rec["feature"], rec["counts"])


def fstats2fmap(fstats_file, outfeaturemap_file, min_counts=0, sep="\t", skip_lines=1):
    """
    convert feature stats file to feature map
    :param fstats_file:
    :param outfeaturemap_file:
    :param min_counts:
    :param sep:
    :param skip_lines:
    :return:
    """
    unk_fid = 0
    fid = 1
    with open(outfeaturemap_file, "w") as fd:
        fd.write("namespace\tfeature\tfid\n")
        for r in gen_feature_stats(fstats_file, sep=sep, skip_lines=skip_lines):
            if int(r.counts) > min_counts:
                fd.write("{ns}\t{feature}\t{fid}\n".format(ns=r.namespace, feature=r.fid, fid=fid))
                fid += 1
            else:
                fd.write("{ns}\t{feature}\t{fid}\n".format(ns=r.namespace, feature=r.fid, fid=unk_fid))
    print("Number of unique features after filtering reare features = {}".format(fid))


def read_feature_map(filename, sep="\t", skip_lines=1):
    fmap = defaultdict(dict)
    for rec in csvreader(filename, sep=sep, skip_lines=skip_lines):
        fmap[rec["namespace"]][rec["feature"]] = rec["fid"]
    return fmap


def remap_features(infilename, fmap_filename, outfilename, feature_cols, extra_cols=None):
    """
    Remap features according to feature map. Only feature columns present in feature map
    remain in output file.
    :param infilename:
    :param fmap_filename:
    :param features - list of features
    :param outfilename:
    :param min_counts:
    :return:
    """
    extra_cols = extra_cols or None
    fmap = read_feature_map(fmap_filename)
    columns = feature_cols + extra_cols
    with open(outfilename, "w") as fd:
        fd.write("{columns}\n".format(columns="\t".join(columns)))
        for rec in csvreader(infilename, skip_lines=1):
            fvals = []
            for col in feature_cols:
                col_val = " ".join([fmap[col][f] for f in rec[col].split()])
                fvals.append(col_val)
            extra = [str(rec[col]) for col in extra_cols]
            fd.write("{val}\n".format(val="\t".join(fvals + extra)))


def train_test_split(infilename, test_prob=0.1):
    with open(infilename) as fd_in:
        header = fd_in.readline()
        with open(infilename + ".train", "w") as fd_train:
            with open(infilename + ".test", "w") as fd_test:
                fd_train.write(header)
                fd_test.write(header)
                for line in fd_in:
                    if random.random() < test_prob:
                        fd_test.write(line)
                    else:
                        fd_train.write(line)


def preprocess(infilename, outfilename, feature_cols, target_col,
               min_counts=None):
    """
    :param infilename:
    :param outfilename:
    :return:
    """
    fmap_file = "{}.feature_map".format(infilename)
    fstats_file = "{}.feature_stats".format(infilename)

    info("Calculating feature stats")
    calc_feature_stats(infilename, feature_cols, fstats_file)
    info("Creating feature map file")
    fstats2fmap(fstats_file, fmap_file, min_counts=min_counts)
    info("Remapping features")
    remap_features(infilename, fmap_file, outfilename, feature_cols, extra_cols=[target_col])


if __name__ == "__main__":
    logging.basicConfig(
        format='[%(asctime)s] %(levelname).1s %(message)s',
        datefmt="%Y.%m.%d %H:%M:%S",
        stream=sys.stdout,
        level=LOGGING_LEVEL
        )

    infile = sys.argv[1]
    outfile = sys.argv[2]
    min_counts = 5
    feature_cols = ["PageID", "OrderID"]
    target_col= "IsClick"
    #preprocess(infile, outfile, feature_cols, target_col, min_counts=min_counts)
    train_test_split(outfile, test_prob=0.1)