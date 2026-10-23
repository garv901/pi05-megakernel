"""PI05_AB_UMMA debug: launch the abu build at one truncated shape (env NS, NL), report errors. Env: NS, NL, XTRA (extra -D flags)."""
import os, sys, torch
from .common import build_mk_arm, make_geo_inputs, setup_torch
setup_torch()
geo, inputs = make_geo_inputs()
ns, nl = int(os.environ.get("NS", "1")), int(os.environ.get("NL", "1"))
extra = tuple(os.environ.get("XTRA", "-DPI05_AB_UMMA=1").split())
arm = build_mk_arm(geo, inputs, "k64", extra=extra)
print("shape", ns, nl, extra, flush=True)
arm.ext.launch(arm.tensors, arm.actions, arm.ws, ns, nl, arm.no_timing, arm.fuse)
torch.cuda.synchronize()
print("OK", "timeouts", int(arm.ext.fg_timeouts()), int(arm.ext.att_timeouts()), flush=True)
