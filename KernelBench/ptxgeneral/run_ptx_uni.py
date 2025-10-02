# run_ptx_universal.py
import argparse, os, importlib.util, json
import torch

from ptx_universal import launch_from_manifest
# For dry-run LLM order preview
try:
	from ptx_universal import _resolve_scalar_order_with_llm  # type: ignore
except Exception:
	_resolve_scalar_order_with_llm = None  # type: ignore

try:
	torch.set_float32_matmul_precision("high")
except Exception:
	pass


def _import_ref_from_path(path: str):
	spec = importlib.util.spec_from_file_location("ref_mod", os.path.abspath(path))
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)  # type: ignore
	# Require Model + get_inputs (KernelBench-style)
	if not hasattr(mod, "Model") or not hasattr(mod, "get_inputs"):
		raise ValueError("Reference module must define Model and get_inputs()")
	return mod


def main():
	p = argparse.ArgumentParser("Universal PTX evaluator (manifest-driven)")
	p.add_argument("--manifest", required=True, help="Path to JSON manifest (in ptx_local)")
	p.add_argument("--ptx", default=None, help="Override PTX path (else use manifest.ptx.file)")
	p.add_argument("--ref", required=True, help="Path to reference Python (Model + get_inputs)")
	p.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
	p.add_argument("--seed", type=int, default=None)
	p.add_argument("--dry-run", action="store_true", help="Only build and print plan; do not load/launch PTX")
	# LLM ABI options
	p.add_argument("--abi-source", default=None, choices=[None, "heuristic", "llm"], help="Select scalar ABI mapping source: heuristic or llm")
	p.add_argument("--llm-server-type", default=None, help="LLM provider key for src.utils presets: openai, deepseek, anthropic, together, google, sglang, fireworks, sambanova")
	p.add_argument("--llm-model-name", default=None, help="Model name to use with the chosen server")
	# Shared memory override
	p.add_argument("--shared-bytes", type=int, default=None, help="Override dynamic shared memory in bytes (e.g., 16384)")
	args = p.parse_args()

	assert torch.cuda.is_available(), "CUDA device required"
	device = torch.device("cuda")

	if args.seed is not None:
		torch.manual_seed(int(args.seed))

	# Load ref & inputs
	ref_mod = _import_ref_from_path(args.ref)
	ref_inputs = [t.to(device=device).contiguous() for t in ref_mod.get_inputs()]
	model = ref_mod.Model().to(device).eval()

	# Optionally downcast inputs
	if args.dtype:
		dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
		ref_inputs = [t.to(dtype=dtype) for t in ref_inputs]
	else:
		dtype = None

	# For dry run we don't need to run the ref, but running it helps infer shapes
	with torch.no_grad():
		y_ref = model(*ref_inputs)
	if isinstance(y_ref, (list, tuple)):
		y_ref = y_ref[0]

	# Launch or dry-run
	plan = launch_from_manifest(
		args.manifest,
		args.ptx,
		ref_inputs,
		device=device,
		dtype_override=dtype,
		dry_run=args.dry_run,
		abi_source=args.abi_source,
		llm_server_type=args.llm_server_type,
		llm_model_name=args.llm_model_name,
		shared_bytes_override=args.shared_bytes,
	)

	# Print plan info
	print("Device:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))
	print("Entry:", plan.entry)
	print("ABI params:", [f"{p['type']} {p['name']}" for p in plan.abi])
	print("Block:", plan.block, "Grid:", plan.grid, "Shared bytes:", plan.shared_bytes)

	if args.dry_run:
		# Detailed per-ABI preview (apply LLM order mapping if requested)
		try:
			with open(args.manifest, "r", encoding="utf-8") as f:
				man = json.load(f)
		except Exception as e:
			print(f"[dry-run] Could not read manifest: {e}")
			print("[dry-run] Skipping PTX load/launch.")
			return

		# Build scalar mapping
		tensors = (man.get("tensors") or {})
		scalars = (man.get("scalars") or {})
		# LLM order resolution (best effort)
		llm_order = None
		if (args.abi_source or "").lower() == "llm" and _resolve_scalar_order_with_llm is not None:
			# We need a PTX text to parse entry/abi for resolver; reuse the manifest's ptx reference if available
			ptx_path = args.ptx
			if not ptx_path:
				ptx_path = (man.get("ptx", {}) or {}).get("file") or man.get("ptx_path")
			try:
				ptx_text_src = ""
				if ptx_path and os.path.exists(ptx_path):
					with open(ptx_path, "r", encoding="utf-8") as f:
						ptx_text_src = f.read()
				# Use plan.abi/entry already computed for the same manifest
				llm_order = _resolve_scalar_order_with_llm(
					server_type=args.llm_server_type,
					model_name=args.llm_model_name,
					entry=plan.entry,
					abi=plan.abi,
					manifest_scalars=scalars,
					manifest_tensors=tensors,
					ptx_text=ptx_text_src,
				)
			except Exception:
				llm_order = None

		# Prepare scalar binding strategy
		print("[dry-run] ABI preview (concrete values/shapes):")
		scalar_idx = 0
		for i, p in enumerate(plan.abi):
			t = p["type"].lstrip(".").lower()
			name = p["name"]
			if t in ("u64", "s64", "b64"):
				# Pointer preview
				# Try to map A/B/C by common names
				ptr_name = None
				for candidate in ("A_ptr","B_ptr","C_ptr","out_ptr","dst_ptr","y_ptr"):
					if candidate in tensors and candidate.lower()[0] == name.lower()[0]:
						ptr_name = candidate
						break
				# Fallback: first available
				if not ptr_name:
					for candidate in ("A_ptr","B_ptr","C_ptr","out_ptr","dst_ptr","y_ptr"):
						if candidate in tensors:
							ptr_name = candidate; break
				meta = tensors.get(ptr_name or "", {})
				print(f"  [{i:02d}] {p['type']} {name} -> ptr {ptr_name or '<unmapped>'} shape={meta.get('shape')} dtype={meta.get('dtype')}")
			else:
				# Scalar preview
				if llm_order and scalar_idx < len(llm_order):
					sym = llm_order[scalar_idx]
					val = scalars.get(sym, 0)
					print(f"  [{i:02d}] {p['type']} {name} -> {sym}={val}")
				else:
					# Fallback heuristic
					preferred = [
						"M","N","K","stride_am","stride_ak","stride_bk","stride_bn","stride_cm","stride_cn",
						"BLOCK_M","BLOCK_N","BLOCK_K","GROUP_SIZE_M",
					]
					sym = next((k for k in preferred if k in scalars), None)
					val = scalars.get(sym, 0) if sym else 0
					print(f"  [{i:02d}] {p['type']} {name} -> {sym or '<scalar unmapped>'}={val}")
				scalar_idx += 1

		print("[dry-run] Skipping PTX load/launch.")
		return

	# Pick the output to compare:
	if plan.outputs:
		y_ptx = plan.outputs[0]
	else:
		raise RuntimeError("No outputs captured. Ensure the manifest marks an output pointer name (e.g., 'C_ptr', 'out_ptr') in tensors.")

	# Compare
	if (y_ref.dtype in (torch.float16, torch.bfloat16)) or (args.dtype in ("float16", "bfloat16")):
		rtol = 1e-3; atol = 1e-3
	else:
		rtol = 1e-6; atol = 1e-6

	same_shape = tuple(y_ptx.shape) == tuple(y_ref.shape)
	max_err = (y_ptx - y_ref).abs().max().item() if same_shape else float("inf")
	ok = same_shape and torch.allclose(y_ptx, y_ref, rtol=rtol, atol=atol)

	print("Output shape:", tuple(y_ptx.shape), "Ref shape:", tuple(y_ref.shape))
	print("max_err:", max_err)
	print(f"allclose (rtol={rtol}, atol={atol}):", ok)
	print("✅ Pass" if ok else "❌ Mismatch")


if __name__ == "__main__":
	main()
