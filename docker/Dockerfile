FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        python3 \
        python3-pip \
        python3-dev \
        python3-numpy \
        python3-pil \
        usbutils \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# Coral Edge TPU apt repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add -

# Edge TPU runtime (std = cooler/stable) + PyCoral (includes compatible tflite-runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libedgetpu1-std \
        python3-pycoral \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

CMD ["bash"]
