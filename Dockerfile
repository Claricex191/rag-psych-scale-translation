# Single image running all three processes for the RAG scale-translation
# service: server_forward.py (8000), server_backward.py (8001), and the
# Streamlit UI (8501, the only port that should be exposed publicly).
#
# On Linux, pip's `torch` wheel bundles its own CUDA runtime libraries, so a
# plain Python base image works fine as long as the host (the GPU pod) has an
# NVIDIA driver and the container is run with GPU access (RunPod does this
# automatically for GPU pod templates).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# poppler-utils: required by pdf2image.convert_from_path (PDF -> page images)
# curl: used by entrypoint.sh to wait on the backend servers' /health
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements-docker.txt = requirements.txt minus rpy2, which needs a system R
# install to build from source and is only used by embed_EGA.py/client_bidirectional.py
# (offline eval tooling, not part of the web service — server_forward.py,
# server_backward.py, streamlit_app.py, and rag_client.py never import it).
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# ColPali (~3B params) is NOT baked into the image at build time — loading it
# needs several GB of RAM/VRAM, which can OOM a constrained build machine (this
# was tested and confirmed locally). Instead its weights are cached on the
# persistent volume via HF_HOME, so they only download once (first request
# after a fresh volume) and survive pod stop/restart from then on.
ENV HF_HOME=/runpod-volume/hf-cache

COPY server_forward.py server_backward.py streamlit_app.py rag_client.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# 8000/8001 (server_forward.py/server_backward.py) are intentionally NOT
# exposed here — entrypoint.sh binds them to 127.0.0.1 only, and only 8501
# should be declared as an exposed HTTP port in the RunPod pod template.
EXPOSE 8501

CMD ["./entrypoint.sh"]
