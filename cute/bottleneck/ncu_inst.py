# ncu source CSV + lineinfo SASS -> pickle list of SASS instructions (addr, inline chain, text, exec, stall samples).  python3 ncu_inst.py ncu_source_sass.csv kernel.lineinfo.sass OUT.pkl
import csv, re, sys, collections
sass_csv, lis = sys.argv[1], sys.argv[2]
amap={}; chain=[]
for l in open(lis):
    m=re.search(r'//## File "([^"]*)", line (\d+)(.*)', l)
    if m:
        f=m.group(1).rsplit("/",1)[1]+":"+m.group(2)
        if "inlined at" not in m.group(3): chain=[f]
        else: chain.append(f)
        continue
    m=re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?);', l)
    if m: amap[int(m.group(1),16)]=(list(chain), m.group(2).strip())
rows=list(csv.reader(open(sass_csv))); hdr=rows[1]; data=rows[2:]; ix={h:i for i,h in enumerate(hdr)}
base=int(data[0][ix["Address"]],16)
def num(r,k):
    try: return float(r[ix[k]].replace(",","") or 0)
    except: return 0.0
stalls=[h for h in hdr if h.startswith("stall_") and "Not Issued" not in h]
inst=[]
for r in data:
    a=int(r[ix["Address"]],16)-base
    ch,txt=amap.get(a,([],r[ix["Source"]]))
    inst.append(dict(a=a,ch=ch,txt=txt,ex=num(r,"Instructions Executed"),thr=num(r,"Thread Instructions Executed"),
        s=num(r,"Warp Stall Sampling (All Samples)"),st={h[6:]:num(r,h) for h in stalls},l2loc=num(r,"L2 Theoretical Sectors Local")))
import pickle; pickle.dump(inst,open(sys.argv[3],"wb"))
