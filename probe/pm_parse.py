"""Parse ncu PmSampling raw CSV cells ('mean (v1; v2; ...)') into numpy series."""
import csv, re
import numpy as np
csv.field_size_limit(1 << 30)

def load(path):
    rows = list(csv.reader(open(path)))
    h = rows[0]; r = rows[2]
    out = {}
    for c, v in zip(h, r):
        m = re.match(r"^\s*([-\d.eE+]*)\s*\((.*)\)\s*$", v, re.S)
        if m:
            try:
                out[c.split(".TriageCompute.")[-1]] = np.array([float(x) for x in m.group(2).split(";") if x.strip() != ""])
            except ValueError:
                pass
    meta = dict(zip(h, r))
    return out, meta

if __name__ == "__main__":
    import sys
    s, meta = load(sys.argv[1])
    for k, v in s.items():
        print(k, len(v), round(float(v.mean()), 3))
    print({k: meta[k] for k in meta if "uration" in k or "nterval" in k or "ample" in k})
