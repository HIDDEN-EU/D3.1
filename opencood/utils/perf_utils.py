"""
Lightweight profiler for timing and GPU-memory measurement of model
sub-components during inference.

Usage
-----
    profiler = PerfProfiler(use_cuda=torch.cuda.is_available())

    # Hook direct-child modules whose names match the given prefixes.
    profiler.hook_model(model, prefixes=('encoder_', 'backbone_', 'aligner_'))

    # Wrap a method that is called *directly* (bypasses nn.Module.__call__).
    profiler.patch_callable(model.pyramid_backbone, 'forward_collab', 'fusion')

    # Wrap a CPU-bound callable (wall-clock timing, no GPU memory tracking).
    profiler.patch_callable(dataset, 'post_process', 'post_process', cpu_only=True)

    # Call at the warmup boundary to discard warmup samples.
    profiler.reset()

    # Build breakdown lines to append to a perf report.
    lines = profiler.report_lines(e2e_mean_ms, timed_prefixes=('encoder_', 'backbone_', 'aligner_'))
"""

import time
import numpy as np
import torch


class PerfProfiler:
    """Accumulates per-component latency and GPU memory deltas."""

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.sub_timings: dict[str, list[float]] = {}   # seconds
        self.sub_mem:     dict[str, list[float]] = {}   # MB, CUDA only
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def hook_model(self, model, prefixes=('encoder_', 'backbone_', 'aligner_')):
        """Register pre/post forward hooks on direct children of *model* whose
        names start with any entry in *prefixes*.

        Uses named_children() (not named_modules()) so that sub-submodule
        paths like ``encoder_m1.pillar_vfe`` are never matched and timed
        independently of their parent.
        """
        for mod_name, submod in model.named_children():
            if any(mod_name.startswith(p) for p in prefixes):
                pre_h, post_h = self._make_hooks(mod_name)
                self._hooks.append(submod.register_forward_pre_hook(pre_h))
                self._hooks.append(submod.register_forward_hook(post_h))

    def _make_hooks(self, key: str):
        """Return (pre_hook, post_hook) that record into *key*."""
        sub_timings = self.sub_timings
        sub_mem = self.sub_mem

        if self.use_cuda:
            def pre(mod, inp):
                torch.cuda.synchronize()
                mod._perf_mem_before = torch.cuda.memory_allocated()
                evt = torch.cuda.Event(enable_timing=True)
                evt.record()
                mod._perf_start_evt = evt

            def post(mod, inp, out):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                torch.cuda.synchronize()
                sub_timings.setdefault(key, []).append(
                    mod._perf_start_evt.elapsed_time(end) / 1000.0)
                sub_mem.setdefault(key, []).append(
                    (torch.cuda.memory_allocated() - mod._perf_mem_before) / (1024 ** 2))
        else:
            def pre(mod, inp):
                mod._perf_t0 = time.time()

            def post(mod, inp, out):
                sub_timings.setdefault(key, []).append(time.time() - mod._perf_t0)

        return pre, post

    # ------------------------------------------------------------------
    # Method patching
    # ------------------------------------------------------------------

    def patch_callable(self, obj, attr: str, key: str, cpu_only: bool = False):
        """Replace ``obj.attr`` with a timed wrapper that records into *key*.

        Parameters
        ----------
        obj:      Object that owns the callable attribute.
        attr:     Name of the attribute to patch (string).
        key:      Key under which timings are stored in sub_timings / sub_mem.
        cpu_only: If True, always use wall-clock timing (time.time) regardless
                  of CUDA availability.  Use this for CPU-bound callables such
                  as post-processing / NMS where CUDA events measure ~0.
        """
        orig = getattr(obj, attr)
        sub_timings = self.sub_timings
        sub_mem = self.sub_mem

        if not cpu_only and self.use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            def wrapper(*a, **kw):
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                start_evt.record()
                result = orig(*a, **kw)
                end_evt.record()
                torch.cuda.synchronize()
                sub_timings.setdefault(key, []).append(
                    start_evt.elapsed_time(end_evt) / 1000.0)
                sub_mem.setdefault(key, []).append(
                    (torch.cuda.memory_allocated() - mem_before) / (1024 ** 2))
                return result
        else:
            def wrapper(*a, **kw):
                if self.use_cuda:
                    torch.cuda.synchronize()   # drain GPU before CPU timer starts
                t0 = time.time()
                result = orig(*a, **kw)
                sub_timings.setdefault(key, []).append(time.time() - t0)
                return result

        setattr(obj, attr, wrapper)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Discard all accumulated samples (call at the warmup boundary)."""
        self.sub_timings.clear()
        self.sub_mem.clear()

    def remove_hooks(self):
        """Deregister all forward hooks (optional cleanup after inference)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_lines(self, e2e_mean_ms: float,
                     timed_prefixes=('encoder_', 'backbone_', 'aligner_')) -> list[str]:
        """Return a list of formatted breakdown strings to append to a perf report.

        Components are shown in sorted order by key name, then fusion, then
        post_process, then an unaccounted remainder line.
        """
        lines = []
        total_s = 0.0

        # Named components (encoder/backbone/aligner etc.)
        for key in sorted(k for k in self.sub_timings
                          if any(k.startswith(p) for p in timed_prefixes)):
            ts = self.sub_timings[key]
            ms = self.sub_mem.get(key, [])
            if not ts:
                continue
            mean_s = float(np.mean(ts))
            total_s += mean_s
            line = (f"  [{key}] mean (ms): {mean_s*1000:.3f}"
                    f"  std: {float(np.std(ts))*1000:.3f}")
            if ms:
                line += (f"  mem_delta (MB): {float(np.mean(ms)):.1f}"
                         f" \u00b1 {float(np.std(ms)):.1f}")
            lines.append(line)

        # Fusion (GPU-timed)
        if self.sub_timings.get('fusion'):
            ts = self.sub_timings['fusion']
            ms = self.sub_mem.get('fusion', [])
            mean_s = float(np.mean(ts))
            total_s += mean_s
            line = (f"  [fusion] mean (ms): {mean_s*1000:.3f}"
                    f"  std: {float(np.std(ts))*1000:.3f}")
            if ms:
                line += (f"  mem_delta (MB): {float(np.mean(ms)):.1f}"
                         f" \u00b1 {float(np.std(ms)):.1f}")
            lines.append(line)

        # Post-processing (CPU wall-clock)
        if self.sub_timings.get('post_process'):
            ts = self.sub_timings['post_process']
            mean_s = float(np.mean(ts))
            total_s += mean_s
            lines.append(
                f"  [post_process] mean (ms): {mean_s*1000:.3f}"
                f"  std: {float(np.std(ts))*1000:.3f}  (wall-clock, CPU-bound)"
            )

        # Unaccounted remainder
        if total_s > 0:
            lines.append(
                f"  [unaccounted: heads/misc] mean (ms): {e2e_mean_ms - total_s*1000:.3f}"
            )

        return lines
