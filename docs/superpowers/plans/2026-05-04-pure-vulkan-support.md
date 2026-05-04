# Pure Vulkan Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vendor-neutral Docker stack (`Dockerfile.gpu-vulkan` + `docker-compose-gpu-vulkan.yml`) that runs the LLM on Vulkan via llama.cpp and the FastAPI/SNAC side on CPU torch, so the project works on any Vulkan-capable GPU without requiring CUDA or ROCm.

**Architecture:** The FastAPI container installs CPU-only PyTorch wheels (no GPU userspace inside it). The existing code in `tts_engine/speechpipe.py` already falls back to `cpu` when `torch.cuda.is_available()` is false, so no Python changes are needed. The `llama-cpp-server` continues to use the existing `ghcr.io/ggml-org/llama.cpp:server-vulkan` image, with ROCm-specific device mounts (`/dev/kfd`, `/dev/mem`) removed so it runs on any Vulkan-capable host.

**Tech Stack:** Docker, docker compose, Python 3.10 (slim), CPU PyTorch wheels, llama.cpp Vulkan server image.

**Spec:** `docs/superpowers/specs/2026-05-04-pure-vulkan-support-design.md`

---

## File Structure

- **Create:** `Dockerfile.gpu-vulkan` — Python slim base, CPU torch, runs the FastAPI app.
- **Create:** `docker-compose-gpu-vulkan.yml` — three services (orpheus-fastapi, llama-cpp-server, model-init), no ROCm device mounts.
- **Modify:** `README.md` — add a "Pure Vulkan" run command alongside the existing CUDA/ROCm/CPU sections.
- **Not changed:** `app.py`, `tts_engine/*`, `requirements.txt`, other Dockerfiles/compose files.

---

## Task 1: Create `Dockerfile.gpu-vulkan`

**Files:**
- Create: `Dockerfile.gpu-vulkan`

- [ ] **Step 1: Write the Dockerfile**

Create `Dockerfile.gpu-vulkan` with the following exact contents:

```dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    libjpeg-dev \
    build-essential \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/outputs /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt ./requirements.txt

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    USE_GPU=false

EXPOSE 5005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
```

- [ ] **Step 2: Build the image to verify it succeeds**

Run:
```bash
docker build -f Dockerfile.gpu-vulkan -t orpheus-fastapi:vulkan-test .
```
Expected: Build completes without errors. Final layer tagged `orpheus-fastapi:vulkan-test`.

- [ ] **Step 3: Verify torch is installed and reports CPU**

Run:
```bash
docker run --rm orpheus-fastapi:vulkan-test python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expected output (version may differ): `torch 2.x.x cuda False`

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.gpu-vulkan
git commit -m "feat: add Dockerfile.gpu-vulkan for vendor-neutral Vulkan stack"
```

---

## Task 2: Create `docker-compose-gpu-vulkan.yml`

**Files:**
- Create: `docker-compose-gpu-vulkan.yml`

- [ ] **Step 1: Write the compose file**

Create `docker-compose-gpu-vulkan.yml` with the following exact contents:

```yaml
services:
  orpheus-fastapi:
    container_name: orpheus-fastapi
    image: orpheus-tts-fastapi-server-orpheus-fastapi:latest
    build:
      context: .
      dockerfile: Dockerfile.gpu-vulkan
    ports:
      - "5005:5005"
    env_file:
      - .env
    environment:
      - ORPHEUS_API_URL=http://llama-cpp-server:5006/v1/completions
    ipc: host
    shm_size: 8g
    restart: unless-stopped
    depends_on:
      llama-cpp-server:
        condition: service_started

  llama-cpp-server:
    image: ghcr.io/ggml-org/llama.cpp:server-vulkan
    ports:
      - "5006:5006"
    volumes:
      - ./models:/models
    env_file:
      - .env
    depends_on:
      model-init:
        condition: service_completed_successfully
    cap_add:
      - SYS_PTRACE
      - CAP_SYS_ADMIN
    security_opt:
      - seccomp=unconfined
    privileged: true
    devices:
      - /dev/dri
    group_add:
      - video
      - render
    ipc: host
    shm_size: 8g
    restart: unless-stopped
    command: >
      -m /models/${ORPHEUS_MODEL_NAME}
      --port 5006
      --host 0.0.0.0
      --n-gpu-layers 29
      --ctx-size ${ORPHEUS_MAX_TOKENS}
      --n-predict ${ORPHEUS_MAX_TOKENS}
      --rope-scaling linear

  model-init:
    image: curlimages/curl:latest
    user: ${UID}:${GID}
    volumes:
      - ./models:/app/models
    working_dir: /app
    command: >
      sh -c '
      if [ ! -f /app/models/${ORPHEUS_MODEL_NAME} ]; then
        echo "Downloading model file..."
        wget -P /app/models https://huggingface.co/lex-au/${ORPHEUS_MODEL_NAME}/resolve/main/${ORPHEUS_MODEL_NAME}
      else
        echo "Model file already exists"
      fi'
    restart: "no"
```

- [ ] **Step 2: Validate compose file syntax**

Run:
```bash
docker compose -f docker-compose-gpu-vulkan.yml config >/dev/null
```
Expected: Exit code 0, no errors. (May print a warning if `.env` is missing — create `.env` from `.env.example` first if so: `cp .env.example .env`.)

- [ ] **Step 3: Commit**

```bash
git add docker-compose-gpu-vulkan.yml
git commit -m "feat: add docker-compose-gpu-vulkan.yml for Vulkan stack"
```

---

## Task 3: End-to-end smoke test

**Files:** none (runtime verification only)

- [ ] **Step 1: Ensure `.env` exists**

Run:
```bash
test -f .env || cp .env.example .env
```
Expected: `.env` file present in project root.

- [ ] **Step 2: Build and start the stack**

Run:
```bash
docker compose -f docker-compose-gpu-vulkan.yml up --build -d
```
Expected: Three services start. `model-init` runs to completion (downloading the GGUF on first run); `llama-cpp-server` and `orpheus-fastapi` reach a running state.

- [ ] **Step 3: Verify FastAPI is using CPU for SNAC**

Run:
```bash
docker logs orpheus-fastapi 2>&1 | grep -i -E "snac|cuda|cpu|gpu" | head -20
```
Expected: Output indicates CPU/no-GPU mode (e.g., a line showing `snac_device` resolved to `cpu`, or "GPU acceleration: No"). No CUDA/ROCm errors.

- [ ] **Step 4: Verify llama.cpp picked a Vulkan device**

Run:
```bash
docker compose -f docker-compose-gpu-vulkan.yml logs llama-cpp-server 2>&1 | grep -i vulkan | head -20
```
Expected: Lines indicating a Vulkan device was selected (e.g., `ggml_vulkan: Found 1 Vulkan devices` or similar). If no Vulkan devices are found, the host is missing a Vulkan ICD or `/dev/dri` access — note this in the README troubleshooting section but do not block the task; the image build itself is what this plan delivers.

- [ ] **Step 5: Issue a TTS request**

Run:
```bash
curl -sS -X POST http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Pure Vulkan smoke test.","voice":"tara","model":"orpheus"}' \
  --output /tmp/orpheus-vulkan-smoke.wav && \
  file /tmp/orpheus-vulkan-smoke.wav
```
Expected: A non-empty WAV file (`file` reports `RIFF (little-endian) data, WAVE audio`). HTTP 200.

- [ ] **Step 6: Tear down**

Run:
```bash
docker compose -f docker-compose-gpu-vulkan.yml down
```
Expected: Containers removed cleanly.

- [ ] **Step 7: No commit for this task** — verification only.

---

## Task 4: Document the Vulkan path in README

**Files:**
- Modify: `README.md` (Docker compose section, around line 134-142)

- [ ] **Step 1: Update the version-list line**

In `README.md`, find this line (around line 114):

```
There are three versions, two for machines that have access to GPU support `docker-compose-gpu.yaml`, `docker-compose-gpu-rocm.yml`  and one for CPU support only: `docker-compose-cpu.yaml`
```

Replace it with:

```
There are four versions: three for machines with GPU support — `docker-compose-gpu.yml` (CUDA), `docker-compose-gpu-rocm.yml` (AMD ROCm), and `docker-compose-gpu-vulkan.yml` (any Vulkan-capable GPU, vendor-neutral) — and one for CPU support only: `docker-compose-cpu.yaml`.
```

- [ ] **Step 2: Add the Vulkan run instructions**

In `README.md`, find this block (around line 134-137):

```
For ROCm GPU support run
```bash
docker compose -f docker-compose-gpu-rocm.yml up
```
```

Immediately after that block (before the "For CPU support run:" block), insert:

```
For Vulkan GPU support (vendor-neutral — Intel, NVIDIA, or AMD without ROCm) run
```bash
docker compose -f docker-compose-gpu-vulkan.yml up
```

In this configuration the LLM runs on the GPU via Vulkan (through `llama.cpp:server-vulkan`) while the SNAC audio decoder runs on CPU torch. This avoids any CUDA or ROCm dependency on the host. Requirements: a working Vulkan ICD on the host and `/dev/dri` accessible to the container — verify with `vulkaninfo` on the host before starting the stack.
```

- [ ] **Step 3: Verify the README renders correctly**

Run:
```bash
grep -n "vulkan" README.md
```
Expected: At least three matches — the version-list line, the "For Vulkan GPU support" heading, and the explanation paragraph mentioning `vulkaninfo`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add pure Vulkan compose instructions to README"
```

---

## Final Verification

- [ ] **Step 1: Confirm all artifacts present**

Run:
```bash
ls -1 Dockerfile.gpu-vulkan docker-compose-gpu-vulkan.yml && \
  grep -c "docker-compose-gpu-vulkan.yml" README.md
```
Expected: Both files listed; grep returns at least `2`.

- [ ] **Step 2: Confirm clean git state**

Run:
```bash
git status
```
Expected: `nothing to commit, working tree clean`.

- [ ] **Step 3: Show the new commits**

Run:
```bash
git log --oneline -5
```
Expected: Top three commits correspond to Task 1 (Dockerfile), Task 2 (compose), Task 4 (README), plus the earlier spec commit.
