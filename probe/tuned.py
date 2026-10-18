"""Persist / inspect the autotuned config of the graph arm (taken from the run-p128 record)."""
import json, os, sys
from .common import TUNED_JSON

SRC = os.path.join(os.path.dirname(__file__), "..", "runs", "20261016T004914Z-p128", "records.jsonl")
# usage: python -m ...probe.tuned [records.jsonl [out.json]]  (defaults: P128 run -> tuned.json)


def extract(src: str = SRC) -> dict:
    for line in open(src):
        r = json.loads(line)
        if r["kind"] == "build" and r.get("arm") == "sota-graph":
            c = r["config"]
            lab = c["attention_backend"]  # triton:bm16_bn64_w8_s1
            import re
            if not lab.startswith("triton:"):   # e.g. "fi:auto" / "sdpa:flash"
                kind, be = lab.split(":")
                return {"source": os.path.relpath(src), "gemm": c["gemm_backends"], "attn": {"kind": kind, "backend": be}}
            kv = {k: int(v) for k, v in re.findall(r"([a-z]+)(\d+)", lab.split(":")[1])}
            attn = {"kind": "triton", "bm": kv["bm"], "bn": kv["bn"], "warps": kv["w"], "stages": kv["s"]}
            return {"source": os.path.relpath(src), "gemm": c["gemm_backends"], "attn": attn}
    raise SystemExit("no sota-graph build record")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    out = sys.argv[2] if len(sys.argv) > 2 else TUNED_JSON
    cfg = extract(src)
    json.dump(cfg, open(out, "w"), indent=1)
    print(cfg)
