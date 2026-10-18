"""Host wrapper of the CuTe megakernel with the MegakernelArm interface (megakernel.py). Compiles once per (knobs, stop)
in-process (the DSL additionally keeps its own JIT cache); the per-launch path is the compiled call only: the kernel zeroes its own
counters on exit (last CTA out, see sync.exit_reset). Knobs.host_zero=True keeps the phase-5 path (one zero_() of the cnt..bar
slice before each launch)."""

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..fixture import Geometry
from ..megakernel import ORDER, parse_variant
from .config import ADIM, NSTEP, NUM_SMS, L, T, Knobs, workspace_layout
from .kernel import Stop, compile_kernel, input_specs

_COMPILED: dict = {}


class CuteArm:
    name = "megakernel-cute"

    def __init__(self, geo: Geometry, variant: str, device: str, knobs: Knobs | None = None, stop: Stop | None = None) -> None:
        if (geo.horizon, geo.action_dim, geo.layers, geo.steps) != (50, 32, 18, 10):
            raise ValueError("the megakernel is specialized for horizon 50, action 32, 18 layers, 10 steps")
        self.geo = geo
        pv = parse_variant(variant)
        if pv["g_noinline"]:
            raise ValueError(f"variant {variant!r}: gn (G noinline) is not supported by the CuTe arm")
        self.knobs = knobs or Knobs(prefix=geo.prefix, att_keys=pv["att_keys"], g_batch=pv["g_batch"])
        if self.knobs.prefix != geo.prefix:
            raise ValueError("knobs.prefix must equal geo.prefix")
        self.stop = stop or Stop()
        self.device = torch.device(device)
        lay = workspace_layout(self.knobs)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.ws = torch.empty(lay["total"][1], dtype=torch.uint8, device=self.device)
                self.nblocks = torch.cuda.get_device_properties(self.device).multi_processor_count
        else:  # CPU-only compile checks (tests): same binding path, no launch
            raw = torch.empty(lay["total"][1] + 256, dtype=torch.uint8, device=self.device)  # CPU malloc is not 256 B aligned
            self.ws = raw[(-raw.data_ptr()) % 256:][:lay["total"][1]]
            self.nblocks = NUM_SMS
        # cnt, ssp, bar are consecutive at the end of the layout: one zero_() over the slice (ssp is rewritten every step)
        # (bind() zeroes only this slice; like the CUDA arm, the other buffers are written before they are read.)
        self._zero = self.ws[lay["cnt"][0]:lay["bar"][0] + lay["bar"][1]]
        self.actions = torch.empty((geo.horizon, geo.action_dim), dtype=torch.float32, device=self.device)
        self.args = None
        self.fn = None
        self.n_steps, self.n_layers = NSTEP, L  # runtime counts (parity.py truncates them for debugging)
        self.config = {"att_keys": self.knobs.att_keys, "g_batch": self.knobs.g_batch, "fuse": self.knobs.fuse, "prefix": geo.prefix,
                       "launches_per_chunk": 1, "stop": self.stop.stage, "impl": "cute"}

    def bind(self, inputs: dict) -> None:
        tensors = []
        for (name, _, _), want in zip(input_specs(self.knobs), ORDER):
            value = inputs[want]
            if not value.is_contiguous():
                raise ValueError(f"{want} must be contiguous")
            tensors.append(value.view(torch.uint8) if value.dtype == torch.bool else value)
        self.tensors = tensors
        self.reset()  # counters must be zero before the first launch (the in-kernel reset keeps them zero afterwards)
        self.args = [from_dlpack(t, assumed_align=16) for t in tensors]
        self.args += [from_dlpack(self.actions, assumed_align=16), from_dlpack(self.ws, assumed_align=256)]
        key = (self.knobs, self.stop)
        if key not in _COMPILED:
            _COMPILED[key] = compile_kernel(self.knobs, self.stop, self.args)
        self.fn = _COMPILED[key]

    def reset(self) -> None:
        """Zero cnt..bar (recovery after an aborted launch; bind() calls it once)."""
        self._zero.zero_()

    def launch(self) -> None:
        if self.knobs.host_zero:
            self._zero.zero_()
        self.fn(*self.args, self.n_steps, self.n_layers, self.nblocks)

    def outputs(self) -> dict:
        return {"actions": self.actions}

    def close(self) -> None:
        self.args = self.tensors = self.fn = None


def cpu_inputs(prefix: int) -> dict:
    """CPU tensors with the shapes, dtypes, strides and contiguity of fixture.make_inputs (derived by running it under
    FakeTensorMode, so reference/pi05.py expert_inputs and fixture._prefix_kv stay the single source); contents are
    uninitialized. For compile checks only."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from ..fixture import make_inputs
    sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **k: None  # fixture.make_inputs synchronizes the (absent) GPU
    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake = make_inputs(Geometry(prefix=prefix), 0, "cpu")
    finally:
        torch.cuda.synchronize = sync
    return {k: torch.empty_strided(tuple(v.shape), v.stride(), dtype=v.dtype) for k, v in fake.items()}


def cpu_arm(knobs: Knobs, stop: Stop | None = None) -> CuteArm:
    """CuteArm on the CPU bound to fixture-shaped tensors: runs the exact bind() path (from_dlpack + compile), no launch."""
    arm = CuteArm(Geometry(prefix=knobs.prefix), f"k{knobs.att_keys}", "cpu", knobs, stop)
    arm.bind(cpu_inputs(knobs.prefix))
    return arm
