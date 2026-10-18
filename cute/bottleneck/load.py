# IKET trace.json -> pickle {(cta, warp): [(start, end, name)]}.  python3 load.py TRACE.json OUT.pkl
import json, pickle, sys, collections
p=sys.argv[1]; out=sys.argv[2]
d=json.load(open(p))
ln=d["launches"][-1]
st=d["stringTable"]; lt=d["locationTable"]
pw=collections.defaultdict(list)
for r in ln["ranges"]:
    nm=st[r["rangeNameIdx"]] if "rangeNameIdx" in r else r["rangeName"]
    for i in dict.fromkeys(r["warpLocIdxs"]):
        loc=lt[i]; pw[(loc["ctaId"][0],loc["warpId"])].append((r["startTs"],r["endTs"],nm))
for v in pw.values(): v.sort()
pickle.dump(dict(pw),open(out,"wb"))
print(len(pw), lt[0])
