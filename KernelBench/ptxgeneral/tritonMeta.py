#!/usr/bin/env python
"""eval_and_export_ptx_modal.py (updated)
Run a Triton kernel via KernelBench's eval function on a Modal GPU, capture PTX,
and emit a runner manifest usable by a generic PTX launcher.

Usage (mode=direct):
  modal run KernelBench/scripts/eval_and_export_ptx_modal.py::main \
    --mode direct \
    --ref    KernelBench/level1/1_Square_matrix_multiplication_.py \
    --triton KernelBench/triton_runs/level1_triton/kernels/level1_problem1_triton.py \
    --gpu H100 \
    --trials 1 \
    --out_dir /vol/ptx_run

Or resolve from a run name (mode=run):
  modal run KernelBench/scripts/eval_and_export_ptx_modal.py::main \
    --mode run \
    --run_name my_run --level 1 --problem_id 1 \
    --gpu H100 --trials 1 --out_dir /vol/ptx_run
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
import json

import modal

# Resolve repo-relative paths based on this file location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(KB_ROOT, "KernelBench", "src")

# -----------------------------------------------------------------------------
# 1.  Modal image & app setup
# -----------------------------------------------------------------------------
CUDA_VERSION = "12.4.0"
FLAVOR       = "devel"
OS_NAME      = "ubuntu22.04"
TAG          = f"{CUDA_VERSION}-{FLAVOR}-{OS_NAME}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "python-dotenv",
        "triton",  # Triton kernels
    )
    .env({"FORCE_REBUILD_V2": datetime.now().isoformat()})
    .add_local_dir(KB_ROOT, remote_path="/root/KernelBench")
    .add_local_dir(SRC_DIR, remote_path="/root/src")
)

APP_NAME = "eval-triton-with-ptx-export"
VOLUME_NAME = "triton-ptx-dumps"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# -----------------------------------------------------------------------------
# 2.  Helpers (PTX parsing / extraction)
# -----------------------------------------------------------------------------
def _parse_ptx(ptx_text: str) -> dict:
    import re
    meta = {"entry": None, "version": None, "target": None, "reqntid": None, "maxntid": None, "params": []}
    m = re.search(r"\.visible\s+\.entry\s+([A-Za-z0-9_]+)\s*\(", ptx_text)
    if m: meta["entry"] = m.group(1)
    m = re.search(r"\.version\s+([0-9.]+)", ptx_text)
    if m: meta["version"] = m.group(1)
    m = re.search(r"\.target\s+([^\n]+)", ptx_text)
    if m: meta["target"] = m.group(1).strip()
    m = re.search(r"\.reqntid\s+(\d+)(?:\s*,\s*(\d+)\s*,\s*(\d+))?", ptx_text)
    if m:
        g = [int(x) if x else 1 for x in m.groups()]
        meta["reqntid"] = g + [1]*(3-len(g))
    m = re.search(r"\.maxntid\s+(\d+)(?:\s*,\s*(\d+)\s*,\s*(\d+))?", ptx_text)
    if m:
        g = [int(x) if x else 1 for x in m.groups()]
        meta["maxntid"] = g + [1]*(3-len(g))
    sig_m = re.search(r"\.visible\s+\.entry\s+[A-Za-z0-9_]+\s*\((.*?)\)\s*\{", ptx_text, re.DOTALL)
    if sig_m:
        meta["params"] = [{"type": t, "name": n}
                          for (t, n) in re.findall(r"\.param\s+(\.\w+)\s+([%$\w\.\[\]]+)", sig_m.group(1))]
    return meta


def _extract_ptx_from_kernel(kernel):
    """Try to pull PTX from Triton kernel object."""
    asm = getattr(kernel, "asm", None)
    if isinstance(asm, dict) and "ptx" in asm:
        return asm["ptx"]
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        for v in cache.values():
            cand = getattr(v, "asm", None)
            if isinstance(cand, dict) and "ptx" in cand:
                return cand["ptx"]
            if isinstance(v, dict) and "asm" in v and "ptx" in v["asm"]:
                return v["asm"]["ptx"]
    return None


def _scan_dir_for_ptx(entry_name: str, root_dir: str):
    """Find newest PTX under root_dir whose .entry matches entry_name."""
    import os, re
    best = None
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith(".ptx"):
                continue
            full = os.path.join(dirpath, f)
            try:
                with open(full, "r", encoding="utf-8") as fh:
                    txt = fh.read()
                m = re.search(r"\.visible\s+\.entry\s+([A-Za-z0-9_]+)\s*\(", txt)
                if m and m.group(1) == entry_name:
                    mt = os.path.getmtime(full)
                    if best is None or mt > best[0]:
                        best = (mt, txt, full)
            except Exception:
                pass
    return (best[1], best[2]) if best else (None, None)


# -----------------------------------------------------------------------------
# 3.  Remote class that evaluates + captures PTX
# -----------------------------------------------------------------------------
@app.cls(gpu="H100", volumes={"/vol": vol})
class TritonEvaluator:
    @modal.method()
    def eval_and_export(
        self,
        ref_src: str,
        triton_src: str,
        gpu_arch: list[str],
        num_perf_trials: int = 1,         # <- fast default
        out_dir: str = "/vol",
        verbose: bool = False,
    ) -> dict:
        """Compile & run the Triton kernel, then copy freshly generated PTX + emit runner manifests."""
        import os, time, shutil, json, glob, torch, re, inspect, hashlib

        # KernelBench imports
        from src.utils import set_gpu_arch as _set_gpu_arch  # noqa: E402
        from src.eval import eval_kernel_against_ref         # noqa: E402
        try:
            from triton.runtime.jit import JITFunction       # Triton ≥ 3
            import triton.language as tl                     # for constexpr
        except Exception:
            JITFunction = None

        _set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        # Ensure dumps go somewhere predictable
        os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
        dump_dir = os.path.join(out_dir, "__triton_dumps") if str(out_dir).startswith("/vol") else "/tmp/triton_dumps"
        os.environ["TRITON_DUMP_DIR"] = dump_dir
        os.makedirs(dump_dir, exist_ok=True)

        # Baseline for "new files since we started"
        start_ts = time.time()

        # ---- Capture grid/args by proxying JITFunction.__getitem__ ----
        captured = {"grid": None, "arg_names": None, "tensors": {}, "scalars": {}, "args": []}
        original_getitem = None

        def _norm_grid(grid_spec, kwargs):
            g = (1, 1, 1)
            if isinstance(grid_spec, int):
                return (grid_spec, 1, 1)
            if isinstance(grid_spec, (tuple, list)):
                t = tuple(grid_spec)
                return t + (1,) * (3 - len(t))
            if callable(grid_spec):
                meta = {}
                # include typical constexpr kwargs (BLOCK_*, num_warps, num_stages)
                for k, v in kwargs.items():
                    if isinstance(v, (int, float)):
                        meta[k] = int(v)
                try:
                    maybe = grid_spec(meta)
                    if isinstance(maybe, (tuple, list)):
                        g = tuple(maybe) + (1,) * (3 - len(maybe))
                except Exception:
                    try:
                        maybe = grid_spec(**meta)
                        if isinstance(maybe, (tuple, list)):
                            g = tuple(maybe) + (1,) * (3 - len(maybe))
                    except Exception:
                        pass
            return g

        if JITFunction is not None:
            original_getitem = JITFunction.__getitem__

            class _KBProxy:
                __slots__ = ("_inner", "_kernel", "_grid_spec")
                def __init__(self, inner, kernel, grid_spec):
                    self._inner = inner
                    self._kernel = kernel
                    self._grid_spec = grid_spec
                def __call__(self, *args, **kwargs):
                    try:
                        g = _norm_grid(self._grid_spec, kwargs)
                        captured["grid"] = g
                        # names for positional args
                        try:
                            sig = inspect.signature(self._kernel.fn)
                            names = [p.name for p in sig.parameters.values()]
                        except Exception:
                            names = [f"arg{i}" for i in range(len(args))]
                        captured["arg_names"] = names
                        ordered = []
                        for name, val in zip(names, args):
                            if isinstance(val, torch.Tensor):
                                meta_t = {"shape": list(val.shape), "dtype": str(val.dtype), "stride": list(val.stride())}
                                captured["tensors"][name] = meta_t
                                ordered.append({"name": name, "kind": "ptr", "tensor_meta": meta_t})
                            elif isinstance(val, (int, float)):
                                sval = int(val)
                                captured["scalars"][name] = sval
                                ordered.append({"name": name, "kind": "scalar", "value": sval})
                            else:
                                ordered.append({"name": name, "kind": "unknown"})
                        # kw scalars (e.g., M,N,K,BLOCK_*)
                        for k, v in kwargs.items():
                            if isinstance(v, (int, float)):
                                captured["scalars"][k] = int(v)
                        captured["args"] = ordered
                    except Exception:
                        pass
                    return self._inner(*args, **kwargs)
                def __getattr__(self, name):
                    return getattr(self._inner, name)

            def new_getitem(self, grid_spec):
                inner = original_getitem(self, grid_spec)
                return _KBProxy(inner, self, grid_spec)

            JITFunction.__getitem__ = new_getitem

        # ---- Run the evaluation (compiles & launches the Triton kernel) ----
        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=triton_src,
            verbose=verbose,
            measure_performance=True,
            num_perf_trials=num_perf_trials,
            backend="triton",
            device=device,
        )

        # Restore hook
        if original_getitem is not None:
            JITFunction.__getitem__ = original_getitem

        # ---- Find new PTX files (cache + dump dir) ----
        cache_root = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
        new_ptx_files: list[str] = []
        for root in [cache_root, dump_dir]:
            if not root or not os.path.isdir(root):
                continue
            for ptx_path in glob.glob(f"{root}/**/*.ptx", recursive=True):
                # small slack because of FS timestamps
                if os.path.getmtime(ptx_path) >= start_ts - 1:
                    new_ptx_files.append(ptx_path)
        # de-dup while preserving order
        seen = set()
        ptx_files = []
        for p in new_ptx_files:
            if p not in seen:
                seen.add(p)
                ptx_files.append(p)

        # ---- Copy PTX + sidecars to out_dir ----
        os.makedirs(out_dir, exist_ok=True)
        saved_paths: list[str] = []
        for src_path in ptx_files:
            base = os.path.basename(src_path)
            dst_path = os.path.join(out_dir, base)
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
                saved_paths.append(dst_path)
                # copy cache-produced JSON if present
                side_json = os.path.splitext(src_path)[0] + ".json"
                if os.path.exists(side_json):
                    shutil.copy2(side_json, os.path.join(out_dir, os.path.basename(side_json)))
            except Exception:
                pass

        # ---- Build runner manifest(s) next to each PTX ----
        files_text: dict[str, str] = {}
        runner_manifests: dict[str, dict] = {}

        def _choose_block(ptx_meta, captured_scalars):
            if ptx_meta.get("reqntid"):
                return ptx_meta["reqntid"]
            if ptx_meta.get("maxntid"):
                return ptx_meta["maxntid"]
            nw = int(captured_scalars.get("num_warps", 4))
            return [nw * 32, 1, 1]

        for ptx_dst in saved_paths:
            try:
                ptx_txt = open(ptx_dst, "r", encoding="utf-8").read()
                ptx_meta = _parse_ptx(ptx_txt)
                kname = ptx_meta.get("entry") or os.path.splitext(os.path.basename(ptx_dst))[0]
                block = _choose_block(ptx_meta, captured.get("scalars", {}))
                grid = list(captured.get("grid") or (1, 1, 1))

                # Align captured args to PTX ABI types
                params = ptx_meta.get("params", [])
                captured_args = list(captured.get("args", []))
                ptrs = [a for a in captured_args if a.get("kind") == "ptr"]
                scal = [a for a in captured_args if a.get("kind") == "scalar"]

                def _is_ptr_type(t: str) -> bool:
                    # b64/u64/s64 are pointer-sized on 64-bit ABI
                    t = t.lstrip(".").lower()
                    return t in ("b64", "u64", "s64")

                aligned = []
                for p in params:
                    if _is_ptr_type(p["type"]):
                        aligned.append(ptrs.pop(0) if ptrs else {"name": p["name"], "kind": "ptr"})
                    else:
                        aligned.append(scal.pop(0) if scal else {"name": p["name"], "kind": "scalar", "value": 0})

                # Emit runner manifest
                runner = {
                    "kernel": kname,
                    "ptx": {
                        "file": os.path.basename(ptx_dst),
                        "version": ptx_meta.get("version"),
                        "target": ptx_meta.get("target"),
                        "params_abi": params,
                    },
                    "block": block,
                    "grid": grid,
                    "args": aligned,
                    "scalars": captured.get("scalars", {}),
                    "tensors": captured.get("tensors", {}),
                }
                rname = os.path.splitext(os.path.basename(ptx_dst))[0] + ".runner.json"
                rpath = os.path.join(out_dir, rname)
                with open(rpath, "w", encoding="utf-8") as f:
                    json.dump(runner, f, indent=2)
                runner_manifests[ptx_dst] = runner

                # collect text payloads so the local entrypoint can mirror locally
                files_text[os.path.basename(ptx_dst)] = ptx_txt
                files_text[os.path.basename(rpath)] = json.dumps(runner, indent=2)

                # include cache-sidecar JSON if present in out_dir now
                side_local = os.path.splitext(ptx_dst)[0] + ".json"
                if os.path.exists(side_local):
                    try:
                        files_text[os.path.basename(side_local)] = open(side_local, "r", encoding="utf-8").read()
                    except Exception:
                        pass
            except Exception:
                pass

        # Persist volume changes if writing to /vol
        if str(out_dir).startswith("/vol"):
            vol.commit()

        return {
            "kernel_exec_result": getattr(result, "dict", lambda: result)(),
            "ptx_files": saved_paths,
            "out_dir": out_dir,
            "files_text": files_text,
            "runner_manifests": runner_manifests,
        }


# -----------------------------------------------------------------------------
# 4.  Local entry-point wrapper
# -----------------------------------------------------------------------------
def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _find_ref_path(level: int, problem_id: int) -> str:
    # Support common layouts:
    #   <KB_ROOT>/KernelBench/level{level}
    #   <KB_ROOT>/KernelBench/KernelBench/level{level}
    #   <KB_ROOT>/level{level}
    candidates_roots = [
        os.path.join(KB_ROOT, "KernelBench", f"level{level}"),
        os.path.join(KB_ROOT, "KernelBench", "KernelBench", f"level{level}"),
        os.path.join(KB_ROOT, f"level{level}"),
    ]

    kb_root = None
    for root in candidates_roots:
        if os.path.isdir(root):
            kb_root = root
            break

    if kb_root is None:
        tried = " and\n".join(candidates_roots)
        raise FileNotFoundError(
            f"Reference dir not found. Tried:\n{tried}.\nEnsure the KernelBench reference levels are present."
        )

    candidates = [n for n in os.listdir(kb_root) if n.startswith(f"{problem_id}_") and n.endswith(".py")]
    if not candidates:
        raise FileNotFoundError(f"No ref file for level {level}, problem {problem_id} under {kb_root}")
    candidates.sort()
    return os.path.join(kb_root, candidates[0])

def _find_triton_path(runs_root: str, run_name: str, level: int, problem_id: int) -> str:
    expected = None
    if run_name:
        expected = os.path.join(runs_root, run_name, "kernels", f"level{level}_problem{problem_id}_triton.py")
        if os.path.exists(expected):
            return expected
    pattern = os.path.join(runs_root, "**", "kernels", f"level{level}_problem{problem_id}_triton.py")
    candidates = glob.glob(pattern, recursive=True)
    if run_name:
        filtered = [p for p in candidates if os.path.sep + run_name + os.path.sep in p]
        if len(filtered) == 1: return filtered[0]
        if len(filtered) > 1:
            raise FileNotFoundError(f"Multiple kernels matched under run '{run_name}': {filtered}.")
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(f"Multiple candidate kernels found: {candidates}.")
    raise FileNotFoundError(f"Triton kernel not found. Tried {expected or '(no expected)'} and glob {pattern}")

@app.local_entrypoint()
def main(
    # selection mode
    mode: str = "direct",          # "run" | "direct"
    # mode=direct
    ref: str | None = None,
    triton: str | None = None,
    # mode=run
    run_name: str | None = None,
    runs_root: str = os.path.join(KB_ROOT, "results", "triton_runs"),
    level: int = 1,
    problem_id: int | None = None,
    # NEW: batch controls for mode=run
    start_problem_id: int | None = None,
    end_problem_id: int | None = None,   # inclusive
    first_n: int | None = None,
    # eval controls
    gpu: str = "H100",
    gpu_arch: str | None = None,   # e.g. "Hopper" | "Ampere" (comma-separated)
    trials: int = 1,               # <- fast default
    out_dir: str = "/vol",
    verbose: bool = False,
):
    if mode == "direct":
        if not ref or not triton:
            raise ValueError("mode=direct requires --ref and --triton")
        ref_src = _read_file(ref)
        triton_src = _read_file(triton)
        problems = [(None, ref_src, triton_src)]  # single payload
    elif mode == "run":
        if not run_name:
            raise ValueError("mode=run requires --run_name")

        # Determine problem id(s)
        problem_ids: list[int]
        if problem_id is not None:
            problem_ids = [problem_id]
        elif start_problem_id is not None and end_problem_id is not None:
            if start_problem_id < 1 or end_problem_id < start_problem_id:
                raise ValueError("Invalid start/end problem ids")
            problem_ids = list(range(start_problem_id, end_problem_id + 1))
        elif first_n is not None:
            if first_n < 1:
                raise ValueError("first_n must be >= 1")
            problem_ids = list(range(1, first_n + 1))
        else:
            raise ValueError("Provide either problem_id, start_problem_id+end_problem_id, or first_n for mode=run")

        # Resolve sources per problem
        problems = []
        for pid in problem_ids:
            ref_path = _find_ref_path(level, pid)
            triton_path = _find_triton_path(runs_root, run_name, level, pid)
            if verbose:
                print(f"Resolved [L{level} P{pid}] ref: {ref_path}")
                print(f"Resolved [L{level} P{pid}] triton: {triton_path}")
            problems.append((pid, _read_file(ref_path), _read_file(triton_path)))
    else:
        raise ValueError("mode must be 'run' or 'direct'")

    # If writing to a local dir, mirror outputs from payload later
    structured_out_dir = out_dir

    # Default arch: match common GPUs
    if gpu_arch:
        arch_list = gpu_arch.split(",")
    else:
        arch_list = ["Hopper"] if gpu.upper().startswith("H100") else ["Ampere"]

    summaries = []
    print("🚀  Submitting job(s) to Modal…")

    # Process all problems (single or batch)
    for pid, ref_src, triton_src in problems:
        # Choose per-problem output dir when batching
        per_out_dir = structured_out_dir
        if pid is not None and structured_out_dir:
            # If out_dir points to /vol, keep per-problem structure under it
            per_out_dir = os.path.join(structured_out_dir, f"level{level}_problem{pid}")

        res = TritonEvaluator.with_options(gpu=gpu)().eval_and_export.remote(
            ref_src,
            triton_src,
            arch_list,
            num_perf_trials=trials,
            out_dir=per_out_dir,
            verbose=verbose,
        )

        # Mirror to local if out_dir is local
        if per_out_dir and not str(per_out_dir).startswith("/vol") and isinstance(res, dict) and "files_text" in res:
            os.makedirs(per_out_dir, exist_ok=True)
            for fname, content in res["files_text"].items():
                with open(os.path.join(per_out_dir, fname), "w", encoding="utf-8") as f:
                    f.write(content)

        # Print per-problem summary
        print("\n====== Execution Result" + (f" [P{pid}]" if pid is not None else "") + " ======")
        ke = res.get("kernel_exec_result", {}) if isinstance(res, dict) else {}
        print("Compiled:", ke.get("compiled"))
        print("Correctness:", ke.get("correctness"))
        rt = ke.get("runtime", -1.0)
        if rt and rt > 0:
            print(f"Runtime (us): {rt:.2f}")

        saved = res.get("ptx_files", []) if isinstance(res, dict) else []
        if saved:
            print("\nPTX files saved to:")
            for p in saved:
                print("  •", p)
        else:
            print("No new PTX files detected — did the kernel launch?")

        summaries.append({
            "problem_id": pid,
            "compiled": ke.get("compiled"),
            "correctness": ke.get("correctness"),
            "runtime_us": rt,
            "ptx_files": saved,
            "out_dir": per_out_dir,
        })

    # Final summary
    if len(summaries) > 1:
        print("\n===== Batch Summary =====")
        for s in summaries:
            tag = f"P{s['problem_id']}" if s["problem_id"] is not None else "(direct)"
            ok = s.get("correctness")
            rt = s.get("runtime_us")
            print(f"{tag}: correctness={ok}, runtime_us={rt}")

    print("\nDone.")
