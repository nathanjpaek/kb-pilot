# ptx_universal.py
from __future__ import annotations
import os, re, json, tempfile, importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from typing import Any

# Optional import of LLM utilities (same style as run_triton_generation)
try:
	from src.utils import create_inference_server_from_presets
except Exception:
	create_inference_server_from_presets = None  # type: ignore


try:
	import cupy as cp
except Exception as e:
	raise RuntimeError(
		"CuPy is required. Install a CUDA-matching build, e.g. `pip install cupy-cuda12x`."
	) from e


# =========================
# PTX parsing (robust)
# =========================

def _parse_ptx_params(ptx_text: str, entry_name: str):
	"""Extract .entry(...) parameter list even if attributes appear between ')' and '{'."""
	import re
	m = re.search(rf"(?:\.visible\s+)?\.entry\s+{re.escape(entry_name)}\s*\(", ptx_text)
	if not m:
		return []
	i = m.end() - 1  # at '('
	depth = 0
	j = i
	while j < len(ptx_text):
		c = ptx_text[j]
		if c == '(':
			depth += 1
		elif c == ')':
			depth -= 1
			if depth == 0:
				end = j
				break
		j += 1
	else:
		return []
	sig_blob = ptx_text[i+1:end]
	params = re.findall(r"\.param\s+(\.\w+)\s+([%$\w\.\[\]]+)", sig_blob)
	return [{"type": t, "name": n} for (t, n) in params]


def parse_ptx(ptx_text: str) -> dict:
	import re
	meta = {"entry": None, "version": None, "target": None, "reqntid": None, "maxntid": None, "params": []}
	em = re.search(r"(?:\.visible\s+)?\.entry\s+([A-Za-z_][\w$@.]*)\s*\(", ptx_text)
	if em: meta["entry"] = em.group(1)
	vm = re.search(r"\.version\s+([0-9.]+)", ptx_text)
	if vm: meta["version"] = vm.group(1)
	tm = re.search(r"\.target\s+([^\n]+)", ptx_text)
	if tm: meta["target"] = tm.group(1).strip()
	for tag, key in (("reqntid","reqntid"),("maxntid","maxntid")):
		m = re.search(rf"\.{tag}\s+(\d+)(?:\s*,\s*(\d+)\s*,\s*(\d+))?", ptx_text)
		if m:
			g = [int(x) if x else 1 for x in m.groups()]
			meta[key] = g + [1]*(3-len(g))
	if meta["entry"]:
		meta["params"] = _parse_ptx_params(ptx_text, meta["entry"])
	return meta

# =========================
# Loader / launcher
# =========================

def _sm_target_from_device(dev: Optional[int] = None) -> str:
	if dev is None:
		dev = torch.cuda.current_device()
	major, minor = torch.cuda.get_device_capability(dev)
	return f"sm_{major}{minor}"


def _maybe_patch_target(ptx_text: str, sm_target: Optional[str]) -> str:
	if not sm_target:
		return ptx_text
	return re.sub(r"\.target\s+sm_\d+[a-z]?", f".target {sm_target}", ptx_text)


def _ptx_version_for_cuda(cuda_version_str: str | None) -> str | None:
	"""Map CUDA version (e.g., '12.1') to a compatible PTX ISA version string (e.g., '8.1')."""
	if not cuda_version_str:
		return None
	try:
		parts = cuda_version_str.split(".")
		major = int(parts[0])
		minor = int(parts[1]) if len(parts) > 1 else 0
	except Exception:
		return None
	# Known mapping for CUDA 12.x
	if major == 12:
		map_12 = {0: "8.0", 1: "8.1", 2: "8.2", 3: "8.3", 4: "8.4"}
		return map_12.get(minor, "8.4")
	# Fallbacks
	if major == 11:
		return "7.8"
	return None


# --- replace in ptx_universal.py ---

def _maybe_patch_target(ptx_text: str, sm_target: str | None) -> str:
	# SAFETY: do not rewrite .target automatically; Hopper '90a' needs to stay '90a'.
	return ptx_text


def _device_cc() -> tuple[int,int]:
	import torch
	return torch.cuda.get_device_capability()


def _load_ptx(ptx_path: str, func_name: str | None):
	import re, os, tempfile, cupy as cp, torch

	with open(ptx_path, "r", encoding="utf-8") as f:
		ptx_text = f.read()

	# Parse meta & guard arch sensibly (sm_90a okay on H100: dev_sm==90)
	meta = parse_ptx(ptx_text)
	target = (meta.get("target") or "").split()[0]  # e.g., "sm_90a"
	m = re.search(r"\bsm_(\d+)(a?)", target)
	want_sm = int(m.group(1)) if m else None
	want_a  = bool(m and m.group(2))
	cc = torch.cuda.get_device_capability()
	dev_sm = cc[0] * 10 + cc[1]                 # (9,0) -> 90
	dev_str = f"sm_{cc[0]}{cc[1]}"

	if want_sm is not None and dev_sm < want_sm:
		raise RuntimeError(
			f"PTX targets {target} but device is {dev_str}. "
			"Re-export PTX for this GPU or run on a matching device."
		)
	if want_a and dev_sm != 90:
		raise RuntimeError(
			f"PTX uses Hopper-only '{target}' features, but device is {dev_str}. "
			"Run on H100/GH200 or re-export without 90a-only ops."
		)

	# Do NOT rewrite .target; keep as-is
	patched = ptx_text

	# Write to a temp file and ONLY delete it AFTER get_function()
	with tempfile.NamedTemporaryFile(mode="w", suffix=".ptx", delete=False) as tmp:
		tmp.write(patched)
		tmp_path = tmp.name

	try:
		mod = cp.RawModule(path=tmp_path)
		if func_name is None:
			m = re.search(r"(?:\.visible\s+)?\.entry\s+([A-Za-z_][\w$@.]*)\s*\(", ptx_text)
			if not m:
				raise ValueError("Could not find .entry in PTX")
			func_name = m.group(1)
		kern = mod.get_function(func_name)  # <- load happens here; tmp must still exist
		return kern, func_name, ptx_text
	finally:
		try:
			os.remove(tmp_path)
		except OSError:
			pass



def _np_for_ptxtype(pt: str):
	"""Map .u32/.s32/.u64/.f32/.f64 -> numpy dtype constructor."""
	t = pt.lstrip(".").lower()
	if t == "u32": return np.uint32
	if t == "s32": return np.int32
	if t in ("u64", "b64", "s64"): return np.uint64  # values only used if we ever pass raw ints
	if t == "f32": return np.float32
	if t == "f64": return np.float64
	# Default to 32-bit unsigned
	return np.uint32


def _torch_dtype_from_string(s: str) -> torch.dtype:
	s = s.lower()
	if "float16" in s or "half" in s: return torch.float16
	if "bfloat16" in s: return torch.bfloat16
	if "float32" in s: return torch.float32
	if "float64" in s or "double" in s: return torch.float64
	if "int64" in s or "long" in s: return torch.int64
	if "int32" in s: return torch.int32
	return torch.float32


def _to_cupy(t: torch.Tensor) -> "cp.ndarray":
	from torch.utils.dlpack import to_dlpack
	return cp.from_dlpack(to_dlpack(t))


@dataclass
class LaunchPlan:
	block: Tuple[int, int, int]
	grid: Tuple[int, int, int]
	shared_bytes: int
	args: tuple
	outputs: List[torch.Tensor]
	abi: List[Dict[str, str]]
	entry: str


def _debug_dump(
	*, entry: str, ptx_meta: dict, manifest: dict,
	abi: list[dict[str, str]],
	block: tuple[int,int,int], grid: tuple[int,int,int], shared_bytes: int,
	scalars: dict[str, Any],
	tensors_meta: dict[str, dict],
	pointer_arrays: dict[str, "cp.ndarray"],
	argv: list[object],
):
	import numpy as np
	print("\n========== PTX LAUNCH DEBUG ==========")
	print(f"Entry: {entry}")
	if ptx_meta:
		print(f"PTX .version: {ptx_meta.get('version')}  .target: {ptx_meta.get('target')}")
		if ptx_meta.get("reqntid"): print(f"PTX .reqntid: {tuple(ptx_meta['reqntid'])}")
		if ptx_meta.get("maxntid"): print(f"PTX .maxntid: {tuple(ptx_meta['maxntid'])}")
	print(f"Block: {block}  Grid: {grid}  Shared bytes: {shared_bytes}")
	print("\nABI params (in order):")
	for i, p in enumerate(abi):
		print(f"  [{i:02d}] {p['type']:>4} {p['name']}")
	print("\nScalars (from manifest):")
	for k, v in scalars.items():
		print(f"  - {k}: {v}")
	print("\nTensors meta (from manifest):")
	for n, m in tensors_meta.items():
		print(f"  - {n}: shape={m.get('shape')} stride={m.get('stride')} dtype={m.get('dtype')}")
	print("\nPointer bindings:")
	for n, arr in pointer_arrays.items():
		try:
			addr = int(arr.data.ptr)
			print(f"  - {n}: addr=0x{addr:016x}")
		except Exception:
			print(f"  - {n}: <no addr?> {type(arr)}")
	print("\nARGV (what will be passed to kernel) [len=", len(argv), "]", sep="")
	for i, v in enumerate(argv):
		if hasattr(v, "dtype"):    # likely numpy scalar
			print(f"  [{i:02d}] numpy {v.dtype} -> {int(v)}")
		elif isinstance(v, (int,)):
			print(f"  [{i:02d}] int -> {v}")
		else:
			# CuPy passes raw pointers as np.uint64; we print hex nicely
			try:
				vv = int(v)
				print(f"  [{i:02d}] ptr -> 0x{vv:016x}")
			except Exception:
				print(f"  [{i:02d}] {type(v)} -> {v}")
	print("======================================\n")

# ---------------- LLM scalar order resolver (optional wiring) ----------------

def _resolve_scalar_order_with_llm(
	*,
	server_type: Optional[str],
	model_name: Optional[str],
	entry: str,
	abi: List[Dict[str, str]],
	manifest_scalars: Dict[str, Any],
	manifest_tensors: Dict[str, Dict[str, Any]],
	ptx_text: str,
) -> Optional[object]:
	# Ensure utils import is available; try lazy import if needed
	global create_inference_server_from_presets
	if create_inference_server_from_presets is None:
		print("[llm-abi] create_inference_server_from_presets is None; attempting lazy import...")
		try:
			import importlib, sys
			print("[llm-abi] CWD:", os.getcwd())
			print("[llm-abi] PYTHONPATH:", os.environ.get("PYTHONPATH"))
			print("[llm-abi] sys.path head:", sys.path[:5])
			# First try normal import
			mod = importlib.import_module("src.utils")
			create_inference_server_from_presets = getattr(mod, "create_inference_server_from_presets", None)
			print("[llm-abi] Lazy import success:", bool(create_inference_server_from_presets))
		except Exception as ie:
			print("[llm-abi] Lazy import failed:", repr(ie))
			# Fallback: search for src/utils.py under /workspace and import by path
			try:
				utils_path = None
				for root, _dirs, files in os.walk("/workspace"):
					if "utils.py" in files and os.path.basename(root) == "src":
						candidate = os.path.join(root, "utils.py")
						if os.path.exists(candidate):
							utils_path = candidate
							break
				print("[llm-abi] Fallback utils path:", utils_path)
				if utils_path:
					import importlib.util as _ilu
					spec = _ilu.spec_from_file_location("src.utils", utils_path)
					module = _ilu.module_from_spec(spec)
					assert spec and spec.loader
					spec.loader.exec_module(module)  # type: ignore
					create_inference_server_from_presets = getattr(module, "create_inference_server_from_presets", None)
					print("[llm-abi] Fallback import success:", bool(create_inference_server_from_presets))
				else:
					print("[llm-abi] Could not find src/utils.py under /workspace")
			except Exception as ie2:
				print("[llm-abi] Fallback import failed:", repr(ie2))
				create_inference_server_from_presets = None

	if not server_type or not model_name or create_inference_server_from_presets is None:
		print("[llm-abi] Skipping LLM mapping (missing server_type/model_name or utils)")
		return None
	# Build a concise prompt asking for scalar order only
	scalar_params = [p for p in abi if p["type"].lstrip(".").lower() not in ("u64","s64","b64")]
	ptr_params = [p for p in abi if p["type"].lstrip(".").lower() in ("u64","s64","b64")]
	ptx_sig_lines = [f"{p['type']} {p['name']}" for p in abi]
	sc_keys = list(manifest_scalars.keys())
	tensor_summ = {
		name: {"shape": meta.get("shape"), "stride": meta.get("stride"), "dtype": meta.get("dtype")}
		for name, meta in (manifest_tensors or {}).items()
	}
	sys_prompt = (
		"You are a tool that returns ONLY a JSON object. "
		"Do not include any prose, code fences, or extra text."
	)
	user_prompt = (
		"Return a single JSON object with this schema: {\"scalar_order\": [string, ...], \"pointer_order\": [string, ...]}\n"
		f"The scalar_order length must equal the number of scalar ABI params ({len(scalar_params)}).\n"
		f"The pointer_order length must equal the number of pointer ABI params ({len(ptr_params)}).\n"
		"pointer_order entries must reference manifest tensor names (e.g., A_ptr, B_ptr, C_ptr) or one of [\"out_ptr\", \"dst_ptr\", \"y_ptr\"].\n"
		"If uncertain, return {\"scalar_order\": [], \"pointer_order\": []}.\n"
		"Output must be between <JSON> and </JSON> tags with nothing else.\n\n"
		"PTX entry: " + entry + "\n"
		"PTX ABI (in order):\n" + "\n".join(ptx_sig_lines) + "\n\n"
		"Manifest.scalars keys: " + ", ".join(sc_keys) + "\n"
		"Manifest.tensors (shape/stride/dtype):\n" + json.dumps(tensor_summ, indent=2) + "\n\n"
		"<JSON>{\n  \"scalar_order\": [\n    \"M\", \"N\", \"K\", \"stride_am\", \"stride_bk\", \"stride_cm\"\n  ],\n  \"pointer_order\": [\n    \"A_ptr\", \"B_ptr\", \"C_ptr\"\n  ]\n}</JSON>\n"
	)
	# Prefer reasoning for OpenAI o* models
	reasoning_kwargs = {}
	try:
		if (server_type or "").lower() == "openai" and (model_name or "").lower().startswith("o"):
			reasoning_kwargs = {"is_reasoning_model": True, "reasoning_effort": "high"}
	except Exception:
		pass

	inference = create_inference_server_from_presets(
		server_type=server_type,
		model_name=model_name,
		greedy_sample=True,
		verbose=True,
		max_tokens=8192,
		**reasoning_kwargs,
	)
	try:
		full_prompt = sys_prompt + "\n\n" + user_prompt
		resp = inference(full_prompt)
		print("[llm-abi] Raw response:")
		print(resp)
		# Prefer <JSON>...</JSON>
		import re as _re
		mjson = _re.search(r"<JSON>([\s\S]*?)</JSON>", resp)
		raw = mjson.group(1).strip() if mjson else None
		if not raw:
			# Fallback to first {...} block
			m = _re.search(r"\{[\s\S]*\}", resp)
			raw = m.group(0) if m else ""
		if not raw:
			raise ValueError("Empty response; no JSON found")
		obj = json.loads(raw)
		order = obj.get("scalar_order")
		porder = obj.get("pointer_order")
		print("[llm-abi] Parsed scalar_order:", order)
		print("[llm-abi] Parsed pointer_order:", porder)
		return {"scalar_order": order, "pointer_order": porder}
	except Exception as e:
		print(f"[llm-abi] Exception while resolving scalar order: {e}")
		return None


def launch_from_manifest(manifest_path: str,
                         ptx_path: Optional[str],
                         ref_inputs: List[torch.Tensor],
                         *,
                         device: torch.device,
                         dtype_override: Optional[torch.dtype] = None,
                         dry_run: bool = False,
                         abi_source: Optional[str] = None,
                         llm_server_type: Optional[str] = None,
                         llm_model_name: Optional[str] = None,
                         shared_bytes_override: Optional[int] = None) -> LaunchPlan:
    """Read JSON manifest + PTX, construct argv in ABI order, optionally launch, and return outputs/plan.

    If dry_run=True, no PTX is loaded or launched. The returned LaunchPlan contains
    block/grid/shared and ABI info; args/outputs will be empty.
    """
    # ---------------- Load manifest + PTX ----------------
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    # Resolve PTX path from manifest if not provided
    if ptx_path is None:
        fname = (man.get("ptx", {}) or {}).get("file") or man.get("ptx_path") or ""
        if not fname:
            raise ValueError("No PTX path provided and manifest has no ptx.file/ptx_path.")
        base = os.path.dirname(os.path.abspath(manifest_path))
        candidate = os.path.join(base, fname)
        ptx_path = candidate if os.path.exists(candidate) else fname

    func_name = man.get("kernel") or man.get("entry")

    # ---------------- Decide block/grid/shared ----------------
    block = tuple(man.get("block") or [128, 1, 1])  # type: ignore
    if len(block) == 1: block = (block[0], 1, 1)
    grid = tuple(man.get("grid") or [1, 1, 1])  # type: ignore
    if len(grid) == 1: grid = (grid[0], 1, 1)
    shared_bytes = int(man.get("dynamic_smem", 0))
    if shared_bytes_override is not None:
        shared_bytes = int(shared_bytes_override)

    # ---------------- ABI and param metadata ----------------
    if dry_run:
        # Best-effort parse ABI without loading the module
        try:
            with open(ptx_path, "r", encoding="utf-8") as f:
                ptx_text_src = f.read()
        except Exception:
            ptx_text_src = ""
        entry = func_name or (parse_ptx(ptx_text_src).get("entry") if ptx_text_src else "<unknown>") or "<unknown>"
        abi = man.get("ptx", {}).get("params_abi") or (parse_ptx(ptx_text_src).get("params") if ptx_text_src else []) or []
        # If requested, run LLM mapping in dry-run so users can see output
        if (abi_source or "").lower() == "llm":
            _ = _resolve_scalar_order_with_llm(
                server_type=llm_server_type,
                model_name=llm_model_name,
                entry=entry,
                abi=abi,
                manifest_scalars=man.get("scalars", {}) or {},
                manifest_tensors=man.get("tensors", {}) or {},
                ptx_text=ptx_text_src or "",
            )
        return LaunchPlan(
            block=(int(block[0]), int(block[1]), int(block[2])),
            grid=(int(grid[0]), int(grid[1]), int(grid[2])),
            shared_bytes=int(shared_bytes),
            args=tuple(()),
            outputs=[],
            abi=abi,
            entry=entry,
        )

    # Non-dry-run: load PTX and build argv
    kernel, entry, ptx_text = _load_ptx(ptx_path, func_name)
    pmeta = parse_ptx(ptx_text)

    # Prefer manifest.args; fallback to PTX params ABI if manifest sparse.
    abi = man.get("ptx", {}).get("params_abi") or pmeta.get("params") or []
    man_args = man.get("args") or []

    # Build quick lookups
    tensors_meta = man.get("tensors", {})  # name -> {shape,stride,dtype}
    scalars = man.get("scalars", {})       # name -> value

    # Prepare pointer objects: map ABI pointer param names -> cp.ndarray
    def _is_output_name(n: str) -> bool:
        ln = n.lower().rstrip("_ptr")
        return ln in ("c", "out", "output", "y", "dst", "o")

    unused_ref_idxs = list(range(len(ref_inputs)))
    pointer_arrays: Dict[str, cp.ndarray] = {}
    output_tensors: Dict[str, torch.Tensor] = {}

    def _shape_dtype_for_name(n: str):
        meta = tensors_meta.get(n) or {}
        shp = meta.get("shape")
        dt = _torch_dtype_from_string(meta.get("dtype", "torch.float32"))
        return shp, dt

    # 1st pass: try to bind obvious input pointers by exact name/shape
    for p in abi:
        t = p["type"].lstrip(".").lower()
        n = p["name"]
        if t in ("u64", "s64", "b64"):
            shp, _ = _shape_dtype_for_name(n)
            bound = False
            if shp:
                for i in list(unused_ref_idxs):
                    if list(ref_inputs[i].shape) == list(shp):
                        pointer_arrays[n] = _to_cupy(ref_inputs[i].contiguous())
                        unused_ref_idxs.remove(i)
                        bound = True
                        break
            if not bound and unused_ref_idxs and n.lower().startswith(("a","b","x","lhs","rhs")) and not _is_output_name(n):
                i = unused_ref_idxs.pop(0)
                pointer_arrays[n] = _to_cupy(ref_inputs[i].contiguous())

    # 2nd pass: allocate outputs or any remaining pointers
    for p in abi:
        t = p["type"].lstrip(".").lower()
        n = p["name"]
        if t in ("u64", "s64", "b64"):
            if n in pointer_arrays:
                continue
            shp, dt = _shape_dtype_for_name(n)
            torch_dt = dtype_override or dt
            if _is_output_name(n) or shp is None:
                M = int(scalars.get("M", 0)); N = int(scalars.get("N", 0))
                if shp is None and (M and N):
                    shp = [M, N]
                if shp is None:
                    raise ValueError(f"Cannot infer shape for output pointer '{n}'. "
                                     f"Add tensors['{n}'].shape to the manifest or provide M/N/etc.")
                out = torch.empty(tuple(shp), device=device, dtype=torch_dt)
                output_tensors[n] = out
                pointer_arrays[n] = _to_cupy(out)
            else:
                out = torch.empty(tuple(shp), device=device, dtype=torch_dt)
                pointer_arrays[n] = _to_cupy(out)

    # Scalars in ABI order (initial fill from manifest)
    scalar_values: Dict[str, int | float] = {}
    for a in man_args:
        if a.get("kind") == "scalar" and "name" in a:
            scalar_values[a["name"]] = a.get("value", scalars.get(a["name"], 0))
    for p in abi:
        t = p["type"].lstrip(".").lower()
        n = p["name"]
        if t not in ("u64", "s64", "b64"):
            if n not in scalar_values:
                scalar_values[n] = scalars.get(n, 0)

    # Optional: LLM-proposed scalar order mapping
    if (abi_source or "").lower() == "llm":
        order_info = _resolve_scalar_order_with_llm(
            server_type=llm_server_type,
            model_name=llm_model_name,
            entry=entry,
            abi=abi,
            manifest_scalars=scalars,
            manifest_tensors=tensors_meta,
            ptx_text=ptx_text,
        )
        if order_info:
            scalar_order = order_info.get("scalar_order")
            pointer_order = order_info.get("pointer_order")
            if scalar_order:
                # Map scalar ABI positions in order to these manifest scalar names
                scalar_params = [p for p in abi if p["type"].lstrip(".").lower() not in ("u64","s64","b64")]
                for abi_param, symbol in zip(scalar_params, scalar_order):
                    # Assign if available, else 0
                    scalar_values[abi_param["name"]] = scalars.get(symbol, 0)
            if pointer_order:
                # Map pointer ABI positions in order to these manifest pointer names
                ptr_param_names = [p["name"] for p in abi if p["type"].lstrip(".").lower() in ("u64","s64","b64")]
                for abi_name, symbol in zip(ptr_param_names, pointer_order):
                    # Assign if available, else None (meaning no binding)
                    pointer_arrays[abi_name] = pointer_arrays.get(symbol)

    # Heuristic: GEMM-like ABI with 3 pointers + 6 scalars -> map to M,N,K and leading strides (am,bk,cn)
    ptr_param_names = [p["name"] for p in abi if p["type"].lstrip(".").lower() in ("u64","s64","b64")]
    scalar_param_names = [p["name"] for p in abi if p["type"].lstrip(".").lower() not in ("u64","s64","b64")]
    if len(ptr_param_names) == 3 and len(scalar_param_names) == 6 and (abi_source or "heuristic").lower() != "llm":
        def _dim(meta_name: str, idx: int) -> Optional[int]:
            m = tensors_meta.get(meta_name) or {}
            shp = m.get("shape") or []
            return int(shp[idx]) if len(shp) > idx and shp[idx] is not None else None
        M = int(scalars.get("M") or _dim("A_ptr", 0) or _dim("C_ptr", 0) or 0)
        K = int(scalars.get("K") or _dim("A_ptr", 1) or _dim("B_ptr", 0) or 0)
        N = int(scalars.get("N") or _dim("B_ptr", 1) or _dim("C_ptr", 1) or 0)
        def _stride0(meta_name: str, fallback: int) -> int:
            m = tensors_meta.get(meta_name) or {}
            st = m.get("stride") or []
            return int(st[0]) if len(st) >= 1 and st[0] is not None else int(fallback)
        def _stride1(meta_name: str, fallback: int) -> int:
            m = tensors_meta.get(meta_name) or {}
            st = m.get("stride") or []
            return int(st[1]) if len(st) >= 2 and st[1] is not None else int(fallback)
        stride_am = int(scalars.get("stride_am") or _stride0("A_ptr", K if K else 1))
        stride_bk = int(scalars.get("stride_bk") or _stride0("B_ptr", N if N else 1))
        stride_cm = int(scalars.get("stride_cm") or _stride1("C_ptr", 1))
        condensed = [M, N, K, stride_am, stride_bk, stride_cm]
        for abi_name, v in zip(scalar_param_names, condensed):
            scalar_values[abi_name] = v

    argv: List[object] = []
    for p in abi:
        t = p["type"].lstrip(".").lower()
        n = p["name"]
        if t in ("u64", "s64", "b64"):
            arr = pointer_arrays.get(n)
            if arr is None:
                raise ValueError(f"Missing device array for pointer param '{n}'.")
            argv.append(arr)
        else:
            ctor = _np_for_ptxtype(p["type"])
            val = scalar_values.get(n, 0)
            argv.append(ctor(val))

    # ---------------- Launch ----------------
    torch_stream = torch.cuda.current_stream().cuda_stream
    stream = cp.cuda.ExternalStream(torch_stream)
    with stream:
        kernel(grid, block, tuple(argv), shared_mem=shared_bytes)
    torch.cuda.synchronize()

    outputs: List[torch.Tensor] = list(output_tensors.values())
    if not outputs and pointer_arrays:
        last_ptr_name = [p["name"] for p in abi if p["type"].lstrip(".").lower() in ("u64","s64","b64")][-1]
        pass

    return LaunchPlan(
        block=(int(block[0]), int(block[1]), int(block[2])),
        grid=(int(grid[0]), int(grid[1]), int(grid[2])),
        shared_bytes=int(shared_bytes),
        args=tuple(argv),
        outputs=outputs,
        abi=abi,
        entry=entry,
    )
