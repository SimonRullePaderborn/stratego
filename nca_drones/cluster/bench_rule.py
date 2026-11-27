"""Throughput of the lattice rule on this machine.  RUN ON THE CLUSTER ONLY.

    python3 nca_drones/cluster/bench_rule.py --out nca_drones/results/bench/h100.json

It times the local rule (3x3x3 convolution, two 1x1x1 layers, residual state, per-voxel normalisation) for the
lattice shapes the programme needs, without gradient (search) and with back-propagation (learning), in three
numeric formats and two memory layouts.  The number that matters is voxel-iterations per second: it replaces the
unmeasured estimate that the design document uses for its cost model.

The rule timed here is a stand-in with the structure of the old 8x8 rule: its input channels pass through the
3x3x3 convolution at every iteration.  The rule of design version 2 pays for its inputs once per step, so its
throughput is about the figure of the row "24x24 proposed rule" times 241,664 / 131,072 = 1.84 (PROGRAMME.md,
section 10, step 5).

The output file holds no host name, so that it can be committed.

Nothing here imports project code, so it can run before anything else is ported.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    """Same structure as nca.SpaceTimeNCA.step (update="res")."""

    def __init__(self, C, hid, n_in):
        super().__init__()
        self.perc = nn.Conv3d(C + n_in, hid, 3, padding=1)
        self.mid = nn.Conv3d(hid, hid, 1)
        self.out = nn.Conv3d(hid, C, 1)

    def forward(self, s, x, amp_dtype):
        # The state stays in float32 (it is a residual accumulator); only the convolutions use the reduced format.
        mu = s.mean(1, keepdim=True)
        z = (s - mu) / (s.var(1, keepdim=True, unbiased=False) + 1e-3).sqrt()
        with torch.autocast(s.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            h = F.relu(self.perc(torch.cat([z, x], 1)))
            d = self.out(F.relu(self.mid(h)))
        return s + d.float()


AMP = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

# name, state channels, hidden width, clamped input channels, slices, height, width, batch for search, batch for learning
SHAPES = [
    ("8x8 rule as trained", 16, 64, 3, 21, 8, 8, 4096, 256),
    ("24x24 proposed rule", 32, 128, 32, 21, 24, 24, 512, 16),
    ("24x24 wider rule", 48, 192, 32, 21, 24, 24, 256, 8),
]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def one_arm(dev, C, hid, n_in, B, T, H, W, iters, grad, amp, channels_last, compiled):
    torch.manual_seed(0)
    rule = Rule(C, hid, n_in).to(dev)
    s = torch.zeros(B, C, T, H, W, device=dev)
    x = torch.randn(B, n_in, T, H, W, device=dev)
    if channels_last:
        rule = rule.to(memory_format=torch.channels_last_3d)
        s = s.contiguous(memory_format=torch.channels_last_3d)
        x = x.contiguous(memory_format=torch.channels_last_3d)
    step = torch.compile(rule) if compiled else rule
    amp_dtype = AMP[amp]

    def unroll(n):
        st = s
        if grad:
            for _ in range(n):
                st = step(st, x, amp_dtype)
            st.pow(2).mean().backward()
            rule.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                for _ in range(n):
                    st = step(st, x, amp_dtype)

    for _ in range(2):                      # warm-up: cuDNN algorithm choice, compilation, allocator
        unroll(2)
        sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    unroll(iters)
    sync(dev)
    dt = time.perf_counter() - t0
    voxel_it = B * T * H * W * iters
    macs = 27 * (C + n_in) * hid + hid * hid + hid * C
    return {
        "seconds": dt,
        "voxel_it_per_s": voxel_it / dt,
        "tflops": 2 * macs * voxel_it / dt / 1e12,
        "macs_per_voxel_it": macs,
        "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="bench_rule.json")
    ap.add_argument("--iters", type=int, default=16, help="rule iterations timed per arm")
    ap.add_argument("--compile", type=int, default=0, help="1: also time torch.compile arms (slow to start)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies all batch sizes (use 0.05 on a CPU)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True      # the fp32 arm is float32 with TF32, as PyTorch documents it
    torch.backends.cudnn.allow_tf32 = True

    meta = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "cpus": os.cpu_count(), "threads": torch.get_num_threads(), "iters": args.iters,
    }
    print(json.dumps(meta))
    rows = []
    for name, C, hid, n_in, T, H, W, b_search, b_learn in SHAPES:
        for grad in (False, True):
            B = max(1, int((b_learn if grad else b_search) * args.scale))
            for amp in ("fp32", "fp16", "bf16"):
                for channels_last in (False, True):
                    for compiled in ((False, True) if args.compile else (False,)):
                        row = {"shape": name, "C": C, "hid": hid, "n_in": n_in, "lattice": [T, H, W], "batch": B,
                               "mode": "learn" if grad else "search", "amp": amp,
                               "channels_last": channels_last, "compiled": compiled}
                        try:
                            row.update(one_arm(dev, C, hid, n_in, B, T, H, W, args.iters, grad, amp, channels_last, compiled))
                            print(f'{name:22s} {row["mode"]:6s} {amp} cl={int(channels_last)} comp={int(compiled)} B={B:5d} '
                                  f'{row["voxel_it_per_s"] / 1e6:9.1f} M voxel-it/s  {row["tflops"]:7.2f} TFLOPS  '
                                  f'mem {row["peak_mem_gb"] if row["peak_mem_gb"] is None else round(row["peak_mem_gb"], 1)} GB', flush=True)
                        except Exception as e:      # an unsupported format or an out-of-memory arm must not stop the sweep
                            row["error"] = f"{type(e).__name__}: {e}"[:300]
                            print(f'{name:22s} {row["mode"]:6s} {amp} cl={int(channels_last)} comp={int(compiled)} B={B:5d} FAILED {row["error"]}', flush=True)
                            if dev.type == "cuda":
                                torch.cuda.empty_cache()
                        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
