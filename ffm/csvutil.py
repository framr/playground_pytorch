def read_table_meta(filename, sep="\t"):
    meta = {}
    with open(filename) as fd:
        meta = fd.readline().strip()
    if meta[0] == "#":
        meta = meta[1:].strip()
    return dict(enumerate(meta.split(sep)))


def parse_record(rec, meta, sep="\t"):
    return dict((meta[index], val) for index, val in enumerate(rec.strip().split(sep)))


def csvreader(filename, sep="\t", skip_lines=0):
    meta = read_table_meta(filename, sep=sep)
    with open(filename) as fd:
        for _ in range(skip_lines):
            fd.readline()
        for line in fd:
            data = parse_record(line, meta, sep=sep)
            yield data
