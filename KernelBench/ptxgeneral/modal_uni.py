import os
import subprocess
import modal

# A lightweight Modal wrapper for the universal PTX evaluator in `ptxgeneral`.
# It mirrors `ptxeval/modal_app.py` but targets `run_ptx_uni.py` and forwards
# the relevant CLI flags for the universal launcher.

app = modal.App("ptx-universal-wrapper")

# --- Image: CUDA + PyTorch + CuPy, plus your local project mounted at /workspace


    .run_commands(
        # Install PyTorch GPU wheels for cu121
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121"
    )
    # Mount the KernelBench root (parent of this file's directory)
    .add_local_dir("..", "/workspace")
)

# Choose a GPU type you have quota for.
GPU = "H100"


@app.function(image=IMAGE, gpu=GPU, timeout=60 * 20, secrets=[modal.Secret.from_name("openai")])
def run_universal(
    # Paths
    script: str | None = None,  # defaults to ptxgeneral/run_ptx_uni.py
    manifest: str | None = None,
    ptx: str | None = None,  # optional override of manifest.ptx.file
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
    Modal uni calls run_ptx_uni.py and that calls ptx_universal.py 

    REGEX: 

      modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal \
          --manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json \
          --ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py --ptx ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.ptx

    Takes in Manifest, Reference Module, and PTX; and the runs the kernel, outputting whether it's correct or not. THIS ONE WORKS!!

    LLM: 

    modal run ptx-triton-gen/KernelBench/ptxgeneral/modal_uni.py::run_universal \
        --manifest ptx-triton-gen/KernelBench/ptxgeneral/ptx_local/matmul_kernel.runner.json \
        --ref ptx-triton-gen/KernelBench/ptxgeneral/matmul_ref.py --abi-source llm \
        --llm-server-type openai --llm-model-name gpt-4o-2024-08-06

    """
    os.chdir("/workspace")
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
        manifest_path = manifest if os.path.isabs(manifest) else os.path.abspath(manifest)
    if (manifest_path is None) or (not os.path.exists(manifest_path)):
        # Search by filename under /workspace
        target = os.path.basename(manifest_path or manifest or "") if (manifest_path or manifest) else None
        found = None
        for root, _, files in os.walk("/workspace"):
            if target and target in files:
                found = os.path.join(root, target)
                break
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
        ref_path = ref if os.path.isabs(ref) else os.path.abspath(ref)
    if (ref_path is None) or (not os.path.exists(ref_path)):
        # Search by filename under /workspace
        target = os.path.basename(ref_path or ref or "") if (ref_path or ref) else "matmul_ref.py"
        found = None
        for root, _, files in os.walk("/workspace"):
            if target and target in files:
                found = os.path.join(root, target)
                break
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

    # Build command line for run_ptx_uni.py
    cmd = ["python", script_path, "--manifest", manifest_path, "--ref", ref_path]
    if ptx_path is not None:
        cmd += ["--ptx", ptx_path]
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
    print("[ptx-universal] Running:", " ".join(cmd))
    if dry_run:
        print("[ptx-universal] Dry-run mode enabled: PTX will not be loaded/launched.")

    # Environment for subprocess
    env = os.environ.copy()
    # Ensure Python can import `src.utils` from KernelBench
    env_paths = [p for p in ["/workspace", "/workspace/src"] if p]
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