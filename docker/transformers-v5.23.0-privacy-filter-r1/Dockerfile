FROM pytorch/pytorch@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
# The lock is the only change to the base's Python packages, so pip may write to the system site-packages.
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes --break-system-packages \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && pip check
# linux-libc-dev holds kernel UAPI headers, which containers never execute; pinned to the noble-security version.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        linux-libc-dev=6.8.0-149.149 \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = "6.8.0-149.149"
COPY server.py /app/server.py
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
