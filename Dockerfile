FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        cmake \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} \
            --gid ${USER_GID} \
            --create-home \
            --shell /bin/bash \
            ${USERNAME} && \
    python3 -m venv /opt/venv && \
    mkdir -p /workspace && \
    chown -R ${USERNAME}:${USERNAME} /workspace /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

COPY --chown=${USERNAME}:${USERNAME} requirements.txt /workspace/

USER ${USERNAME}

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=${USERNAME}:${USERNAME} . /workspace

CMD ["bash"]