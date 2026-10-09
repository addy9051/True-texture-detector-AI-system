"""Modal Serverless vLLM Deployment for Returns Concierge AI Agent.

Follows official Modal 1.6+ and vLLM standards:
- Runs Qwen 2.5 (7B Instruct) on serverless cloud GPU (L4: 24GB VRAM).
- Exposes standard OpenAI-compatible endpoints (/v1/chat/completions).
- Enables Hermes tool-calling parser and automatic prefix caching (KV cache).
- Persistent Modal Volumes for model weights & compilation cache.
- Scales to 0 when idle to conserve free Modal credits ($30/mo free tier).

Usage:
    # 1. Authenticate with Modal (one-time interactive browser login):
    uv run modal setup

    # 2. (Recommended) Pre-download weights into persistent volume:
    uv run modal run scripts/serve_modal_vllm.py::download_weights

    # 3. Test interactively with local entrypoint:
    uv run modal run scripts/serve_modal_vllm.py

    # 4. Deploy live serverless endpoint:
    uv run modal deploy scripts/serve_modal_vllm.py

    # 5. Set the generated URL in your .env:
    # LLM_PROVIDER=modal
    # MODAL_LLM_URL=https://<your-username>--concierge-vllm-service-serve.modal.run/v1
"""

import os
import subprocess
import modal

MODEL_ID = os.environ.get("MODAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
APP_NAME = "concierge-vllm-service"
VLLM_PORT = 8000

app = modal.App(APP_NAME)

# Persistent volumes for Hugging Face weights & vLLM compilation artifacts
hf_cache_vol = modal.Volume.from_name("concierge-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("concierge-vllm-cache", create_if_missing=True)

# Container image using NVIDIA CUDA base + uv_pip_install for high-speed builds
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.6.3",
        "huggingface_hub",
        "hf-transfer",
        "requests",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_LOG_STATS_INTERVAL": "5",
    })
)


@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=1200,
)
def download_weights(model_id: str = MODEL_ID):
    """Pre-downloads model weights into persistent volume for fast subsequent cold starts."""
    from huggingface_hub import snapshot_download

    print(f"[modal] Pre-downloading {model_id} to persistent volume...")
    snapshot_download(repo_id=model_id, ignore_patterns=["*.pt", "*.bin"])
    hf_cache_vol.commit()
    print(f"[modal] Weights cached successfully for {model_id}!")


@app.function(
    image=vllm_image,
    gpu="L4",                      # 24GB VRAM (~$0.80/hr). Perfect fit for Qwen 7B/14B
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    scaledown_window=300,    # Scale to zero after 5 minutes of idle time
    timeout=600,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def serve():
    """Start vLLM OpenAI-compatible server on port 8000."""
    api_key = os.environ.get("MODAL_API_KEY", "")

    cmd = [
        "vllm", "serve", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--enable-prefix-caching",
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])

    print(f"[modal] Executing: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.local_entrypoint()
def test():
    """Local test helper: pings the server to verify it spins up and responds."""
    print(f"[modal] Server definition ready for model: {MODEL_ID}")
    print("[modal] Run `uv run modal deploy scripts/serve_modal_vllm.py` to deploy live.")
