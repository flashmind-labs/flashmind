# Packages a prebuilt Linux `flsh` binary. The build context is a directory
# holding flsh-<arch>-unknown-linux-gnu, as the release workflow produces:
#
#   docker buildx build -f docker/Dockerfile --platform linux/amd64,linux/arm64 dist
#
#   docker run -it --rm -v "$PWD:/work" -v flsh:/data -e ANTHROPIC_API_KEY ghcr.io/flashmind-labs/flsh

# trixie for glibc 2.41: the binaries are built on Ubuntu 24.04 (glibc 2.39).
FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git less openssh-client procps ripgrep tini \
    && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
RUN --mount=type=bind,target=/tmp/dist \
    case "$TARGETARCH" in \
        amd64) arch=x86_64 ;; \
        arm64) arch=aarch64 ;; \
        *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && install -m 0755 "/tmp/dist/flsh-$arch-unknown-linux-gnu" /usr/local/bin/flsh

RUN useradd --create-home --uid 1000 flsh \
    && mkdir -p /data /work \
    && chown flsh:flsh /data /work \
    && chmod 1777 /data /home/flsh

# All config, sessions and logs live in the /data volume. Set
# RUST_LOG_STDERR=1 for headless runs so `docker logs` shows the log.
ENV FLASHMIND_HOME=/data

USER flsh
WORKDIR /work
VOLUME ["/data"]

ENTRYPOINT ["/usr/bin/tini", "--", "flsh"]
