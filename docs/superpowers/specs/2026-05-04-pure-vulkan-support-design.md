# Pure Vulkan Support — Design

## Goal

Provide a vendor-neutral GPU stack for Orpheus-FastAPI that runs on any Vulkan-capable GPU (Intel, NVIDIA, AMD without ROCm), without requiring CUDA or ROCm userspace inside the FastAPI container.

## Background

The current `docker-compose-gpu-rocm.yml` already runs the LLM on `ghcr.io/ggml-org/llama.cpp:server-vulkan` (Vulkan), but the `orpheus-fastapi` service installs ROCm PyTorch wheels for the SNAC decoder. This couples the whole stack to AMD/ROCm.

We evaluated two third-party "Vulkan PyTorch" projects and rejected both:

- **`ixu2486/pytorch_retryix_backend`** — Windows-only wheels, registers as `privateuseone:0`, would need code changes since the project keys off `torch.cuda.is_available()`.
- **ExecuTorch Vulkan backend** — ahead-of-time only, requires `.pte` export via `VulkanPartitioner`, no eager mode, mobile-focused.

Decision: run torch on **CPU** for the FastAPI/SNAC side. SNAC decoding is small relative to LLM inference, and the existing code already falls back to CPU automatically when `torch.cuda.is_available()` is false (`tts_engine/speechpipe.py:43`).

## Changes

### New file: `Dockerfile.gpu-vulkan`

- Base: `python:3.10-slim` (no GPU userspace required in this container).
- System deps: `libsndfile1`, `ffmpeg`, `portaudio19-dev`, `libjpeg-dev`, plus build tooling needed by pip deps.
- Non-root `appuser` (uid 1001), venv at `/app/venv`, same layout as `Dockerfile.gpu-rocm`.
- Install torch from the CPU index:
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- Then `pip install -r requirements.txt`.
- `ENV USE_GPU=false PYTHONUNBUFFERED=1 PYTHONPATH=/app`.
- `EXPOSE 5005`, `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]`.

### New file: `docker-compose-gpu-vulkan.yml`

Copy of `docker-compose-gpu-rocm.yml` with:

- **`orpheus-fastapi`**: build `Dockerfile.gpu-vulkan`. Drop `privileged`, `cap_add`, `security_opt`, `devices`, `group_add` (no GPU access needed). Keep `ipc: host`, `shm_size: 8g`, `env_file`, `ORPHEUS_API_URL`, `depends_on`, `restart`.
- **`llama-cpp-server`**: image unchanged (`server-vulkan`). Devices reduced from `/dev/kfd, /dev/dri, /dev/mem` to **`/dev/dri` only** (the only one Vulkan needs). Drop the hardcoded `993` numeric group; keep `video` and `render`. Keep `privileged`, `cap_add`, `security_opt`, `ipc`, `shm_size` as-is so behavior matches the rocm compose for users who already have it working.
- **`model-init`**: unchanged.

### `README.md` update

Add a "Pure Vulkan (vendor-neutral GPU)" section under the existing GPU instructions explaining:
- When to use this image (any Vulkan-capable GPU; no CUDA/ROCm install required).
- How to run: `docker compose -f docker-compose-gpu-vulkan.yml up --build`.
- Note that SNAC runs on CPU in this configuration; the LLM runs on the GPU via Vulkan.
- Host requirements: a working Vulkan ICD and `/dev/dri` access (mention `vulkaninfo` as a quick check).

### Files **not** changed

- `app.py`, `tts_engine/*`, `requirements.txt`. Existing CPU fallback in `speechpipe.py` handles the device selection.

## Verification

1. `docker compose -f docker-compose-gpu-vulkan.yml build` succeeds.
2. `docker compose -f docker-compose-gpu-vulkan.yml up` starts both services.
3. FastAPI startup log shows `snac_device = cpu` (or equivalent — verify against actual log line in `speechpipe.py`).
4. llama.cpp server log shows a Vulkan device being selected.
5. A TTS request to `http://localhost:5005` returns audio successfully.

## Out of scope

- Running torch itself on Vulkan (no production-ready path today).
- Changes to the existing ROCm or CUDA Dockerfiles/compose files.
- Performance tuning of SNAC on CPU.
