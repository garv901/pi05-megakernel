"""Launch-side half of splice.retarget_bank4: upload the original module's global data (.nv.global.init, then .nv.global
zeroed) and give the addresses that stand in for the dropped constant-bank-4 slots, in the order of the parameters the
splice added (between the kernel's own parameters and the trace pointer). No gwatch import: runs in either venv."""
from __future__ import annotations

import torch


def upload(meta):
    """(device tensor to keep alive, [address per added parameter])."""
    b = meta.get("bank4") or {"slots": [], "init_hex": ""}
    data = bytes.fromhex(b["init_hex"]) or b"\0"
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    return t, [t.data_ptr() + s["init_off"] for s in b["slots"]]
