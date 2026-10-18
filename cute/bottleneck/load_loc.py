# IKET trace.json -> pickle {"loc": {(cta,warp): (gpc,tpc,sm)}, "r": {(cta,warp): [(start,end,name)]}}.  python3 load_loc.py TRACE.json OUT.pkl
import json, pickle, sys, collections
d=json.load(open(sys.argv[1])); ln=d["launches"][-1]; st=d["stringTable"]; lt=d["locationTable"]
pw=collections.defaultdict(list); loc={}
for l in lt: loc[(l["ctaId"][0],l["warpId"])]=(l["gpcId"],l["tpcId"],l["smId"])
for r in ln["ranges"]:
    nm=st[r["rangeNameIdx"]]
    for i in dict.fromkeys(r["warpLocIdxs"]):
        l=lt[i]; pw[(l["ctaId"][0],l["warpId"])].append((r["startTs"],r["endTs"],nm))
for v in pw.values(): v.sort()
pickle.dump({"loc":loc,"r":dict(pw)},open(sys.argv[2],"wb"))
