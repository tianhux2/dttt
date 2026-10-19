# The training environment for GPU nodes: Python and the locked dependencies of `uv sync --extra cuda`. The code is
# not in the image: tools/docker_run.sh mounts the repository at /workspace (docs/launch.md, section 5).
# jax[cuda13] brings CUDA 13 as wheels; the base image only has to match it for the driver check (driver >= 580).
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# Environment settings. UV_PYTHON_DOWNLOADS=never: the venv must use Ubuntu's python3; a Python downloaded by uv
# would land in /root, which the calling user of tools/docker_run.sh cannot read, so a mismatch fails the build.
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/workspace/src:/workspace" \
    JAX_PLATFORMS=cuda \
    JAX_COMPILATION_CACHE_DIR=/jax_cache

# Install system dependencies: git records the commit in every checkpoint, the verbs libraries let NCCL use
# InfiniBand. safe.directory: git refuses a repository owned by another user (a checkout shared by several users).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    python3 \
    libibverbs1 \
    ibverbs-providers \
    librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace

# uv at the version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

WORKDIR /build

# Copy dependency specifications only to optimize Docker layer caching
COPY pyproject.toml uv.lock .python-version ./

# Pre-install all Python dependencies for CUDA into /opt/venv. The uv cache stays outside the image; --locked
# refuses a uv.lock that no longer matches pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra cuda

WORKDIR /workspace

CMD ["/bin/bash"]
