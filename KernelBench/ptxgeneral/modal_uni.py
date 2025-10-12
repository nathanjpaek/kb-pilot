import os
import subprocess
import modal
from dotenv import load_dotenv

# A lightweight Modal wrapper for the universal PTX evaluator in `ptxgeneral`.
# It mirrors `ptxeval/modal_app.py` but targets `run_ptx_uni.py` and forwards
# the relevant CLI flags for the universal launcher.

app = modal.App("ptx-universal-wrapper")

IMAGE = (
    modal.Image.debian_slim()
    .pip_install([
        "numpy",
        "cupy-cuda12x",
        "openai",
        "anthropic",
        "google-generativeai",
        "together",
        "python-dotenv",
        "transformers",
        "datasets",
        "tqdm",
    ])
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121"
    )
    .add_local_dir(
        "..",
        "/workspace",
        ignore=[
            "**/__pycache__",
            "**/*.pyc",
            "**/.git",
            "**/triton_cache",
            "**/results/**/triton_cache",
            "**/runs/*/triton_cache",
            "**/*.egg-info",
            "**/20250729-091512",
            ".pytest_cache",
            ".mypy_cache",
            "*.cubin",
            "*.so",
        ]
    )
)

GPU = "H100"


@app.function(image=IMAGE, gpu=GPU, timeout=60 * 20)
def run_universal(
    script: str | None = None,  # defaults to ptxgeneral/run_ptx_uni.py
    manifest: str | None = None,
    ptx: str | None = None,  # optional override of manifest.ptx.file
    kernel_meta: str | None = None,  # path to Triton kernel metadata JSON
    ref: str | None = None,  # path to reference module (Model + get_inputs)
    workdir: str | None = None,  # optional working directory
    # Runner args
    dtype: str | None = None,  # one of: None,float32,float16,bfloat16
    seed: int | None = None,
    # Debug/behavior
    strict: bool = True,            # if False, do not raise on non-zero exit
    stream_output: bool = True,     # if True, stream child output live
    cuda_launch_blocking: bool = False,  # if True, set CUDA_LAUNCH_BLOCKING=1
    dry_run: bool = False,          # if True, only print plan; do not load/launch PTX
    # LLM ABI options
    abi_source: str | None = None,          # "heuristic" or "llm"
    llm_server_type: str | None = None,     # e.g., openai, deepseek, anthropic, together
    llm_model_name: str | None = None,      # model id for the chosen server
    # Shared mem override
    shared_bytes: int | None = None,
):
    """Run the universal PTX evaluator inside a GPU container.

    Examples:
      modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal \
          --manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json \
          --ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py

      modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal \
          --manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json \
          --ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py \
          --ptx ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.ptx \
          --kernel-meta ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.json
    """
    os.chdir("/workspace")
    # Load environment variables from a .env file mounted at /workspace
    # Expected to include OPENAI_API_KEY (and any other needed keys)
    try:
        load_dotenv(dotenv_path="/workspace/.env", override=False)
        # Also load from current working directory if present (non-fatal if missing)
        load_dotenv(override=False)
    except Exception:
        pass
    if workdir:
        # Allow caller to set a cwd that may contain the reference module
        wd = workdir if os.path.isabs(workdir) else os.path.abspath(workdir)
        if not os.path.isdir(wd):
            raise NotADirectoryError(f"workdir not found: {workdir}")
        os.chdir(wd)
    else:
        # Auto-detect a directory containing matmul_ref.py if not already present
        if not os.path.exists("matmul_ref.py"):
            found_ref_dir = None
            for root, _dirs, files in os.walk("/workspace"):
                if "matmul_ref.py" in files:
                    found_ref_dir = root
                    break
            if found_ref_dir:
                os.chdir(found_ref_dir)

    # Default script path
    default_script = "ptx-triton-gen/KernelBench/ptxgeneral/run_ptx_uni.py"
    script_path = script or default_script
    if not os.path.exists(script_path):
        # Try basename in CWD
        fallback = os.path.basename(script_path)
        if os.path.exists(fallback):
            script_path = fallback
        else:
            # Search under /workspace
            found = None
            for root, _, files in os.walk("/workspace"):
                if os.path.basename(script_path) in files:
                    found = os.path.join(root, os.path.basename(script_path))
                    break
            if found is None:
                raise FileNotFoundError(f"Could not find runner script: {script_path}")
            script_path = found

    # Resolve manifest path (default to the example runner manifest if present)
    if manifest is None:
        candidate = "ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json"
        manifest_path = candidate if os.path.exists(candidate) else None
    else:
        # Try multiple base directories for relative paths
        if os.path.isabs(manifest):
            manifest_path = manifest
        else:
            # Try relative to: CWD, /workspace, /workspace/kb_triton_ptx
            candidates = [
                os.path.abspath(manifest),
                os.path.join("/workspace", manifest),
                os.path.join("/workspace/kb_triton_ptx", manifest),
            ]
            manifest_path = next((c for c in candidates if os.path.exists(c)), manifest)
    
    if (manifest_path is None) or (not os.path.exists(manifest_path)):
        # Last resort: search by filename under /workspace (warn about ambiguity)
        target = os.path.basename(manifest_path or manifest or "") if (manifest_path or manifest) else None
        found = None
        matches = []
        for root, _, files in os.walk("/workspace"):
            if target and target in files:
                matches.append(os.path.join(root, target))
        if len(matches) > 1:
            print(f"[ptx-universal] Warning: Multiple files named '{target}' found:")
            for m in matches:
                print(f"  - {m}")
            print(f"[ptx-universal] Using: {matches[0]}")
        found = matches[0] if matches else None
        if found is None:
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        manifest_path = found

    # Resolve reference module path (default to matmul_ref.py)
    if ref is None:
        # Prefer local matmul_ref.py in CWD if present
        if os.path.exists("matmul_ref.py"):
            ref_path = os.path.abspath("matmul_ref.py")
        else:
            ref_path = None
    else:
        # Try multiple base directories for relative paths
        if os.path.isabs(ref):
            ref_path = ref
        else:
            candidates = [
                os.path.abspath(ref),
                os.path.join("/workspace", ref),
                os.path.join("/workspace/kb_triton_ptx", ref),
            ]
            ref_path = next((c for c in candidates if os.path.exists(c)), ref)
    
    if (ref_path is None) or (not os.path.exists(ref_path)):
        # Last resort: search by filename under /workspace (warn about ambiguity)
        target = os.path.basename(ref_path or ref or "") if (ref_path or ref) else "matmul_ref.py"
        found = None
        matches = []
        for root, _, files in os.walk("/workspace"):
            if target and target in files:
                matches.append(os.path.join(root, target))
        if len(matches) > 1:
            print(f"[ptx-universal] Warning: Multiple files named '{target}' found:")
            for m in matches:
                print(f"  - {m}")
            print(f"[ptx-universal] Using: {matches[0]}")
        found = matches[0] if matches else None
        if found is None:
            raise FileNotFoundError(f"Reference module not found: {ref or 'matmul_ref.py'}")
        ref_path = found

    # Resolve PTX path (optional override)
    if ptx is None:
        ptx_path = None
    else:
        ptx_path = ptx if os.path.isabs(ptx) else os.path.abspath(ptx)
    if (ptx_path is not None) and (not os.path.exists(ptx_path)):
        # Search by filename under /workspace
        target = os.path.basename(ptx_path)
        found = None
        for root, _, files in os.walk("/workspace"):
            if target and target in files:
                found = os.path.join(root, target)
                break
        if found is None:
            raise FileNotFoundError(f"PTX not found: {ptx}")
        ptx_path = found

    # Resolve kernel metadata JSON path (optional)
    if kernel_meta is None:
        kernel_meta_path = None
    else:
        kernel_meta_path = kernel_meta if os.path.isabs(kernel_meta) else os.path.abspath(kernel_meta)
    if (kernel_meta_path is not None) and (not os.path.exists(kernel_meta_path)):
        # Search by filename under /workspace
        target = os.path.basename(kernel_meta_path)
        found = None
        for root, _, files in os.walk("/workspace"):
            if target and target in files:
                found = os.path.join(root, target)
                break
        if found is None:
            raise FileNotFoundError(f"Kernel metadata JSON not found: {kernel_meta}")
        kernel_meta_path = found

    # Build command line for run_ptx_uni.py
    cmd = ["python", script_path, "--manifest", manifest_path, "--ref", ref_path]
    if ptx_path is not None:
        cmd += ["--ptx", ptx_path]
    if kernel_meta_path is not None:
        cmd += ["--kernel-meta", kernel_meta_path]
    if dtype is not None:
        cmd += ["--dtype", dtype]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if dry_run:
        cmd += ["--dry-run"]
    # LLM flags
    if abi_source is not None:
        cmd += ["--abi-source", abi_source]
    if llm_server_type is not None:
        cmd += ["--llm-server-type", llm_server_type]
    if llm_model_name is not None:
        cmd += ["--llm-model-name", llm_model_name]
    if shared_bytes is not None:
        cmd += ["--shared-bytes", str(shared_bytes)]

    print("[ptx-universal] CWD:", os.getcwd())
    print("[ptx-universal] Script:", script_path)
    print("[ptx-universal] Manifest:", manifest_path)
    print("[ptx-universal] Ref:", ref_path)
    print("[ptx-universal] PTX override:", ptx_path or "<None>")
    print("[ptx-universal] Kernel metadata:", kernel_meta_path or "<None>")
    print("[ptx-universal] Running:", " ".join(cmd))
    if dry_run:
        print("[ptx-universal] Dry-run mode enabled: PTX will not be loaded/launched.")

    # Environment for subprocess
    env = os.environ.copy()
    # Ensure Python can import `src.utils` from KernelBench
    bench_root = os.path.abspath(os.path.join(os.path.dirname(script_path), ".."))
    env_paths = [p for p in [bench_root, os.path.join(bench_root, "src") ] if p]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (existing + (":" if existing else "") + ":".join(env_paths))
    if cuda_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    if stream_output:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        ret = proc.returncode
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        print(proc.stdout)
        ret = proc.returncode

    print(f"[ptx-universal] Exit code: {ret}")
    if strict and ret != 0:
        raise RuntimeError(f"{os.path.basename(script_path)} exited with code {ret}")
    return ret


@app.local_entrypoint()
def main():
    print(
        "Examples:\n"
        "  modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal "
        "--manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json "
        "--ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py\n"
        "  modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal "
        "--manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json "
        "--ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py --ptx ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.ptx\n"
        "  # With LLM ABI mapping:\n"
        "  modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal "
        "--manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json "
        "--ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py --abi-source llm --llm-server-type openai --llm-model-name gpt-4o-2024-08-06\n"
    )
