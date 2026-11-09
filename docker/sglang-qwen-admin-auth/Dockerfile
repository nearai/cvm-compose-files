# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG PATCH_FILE
ARG BUILD_SOURCE_REVISION
ARG SGLANG_SOURCE_REVISION
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      nearai.build.recipe="docker/sglang-qwen-admin-auth/Dockerfile" \
      nearai.build.source_revision="${BUILD_SOURCE_REVISION}" \
      nearai.sglang.source_revision="${SGLANG_SOURCE_REVISION}" \
      nearai.sglang.admin_auth_redaction="v1-upstream-41322-48518"
COPY . /usr/share/nearai/sglang-qwen-admin-auth/
RUN set -eu; \
    test -n "$BASE_IMAGE"; test -n "$BUILD_SOURCE_REVISION"; test -n "$SGLANG_SOURCE_REVISION"; \
    case "$PATCH_FILE" in admin-auth-redaction-v0.5.16.diff|admin-auth-redaction-c4271c3f.diff) ;; *) exit 1 ;; esac; \
    cd /usr/share/nearai/sglang-qwen-admin-auth; sha256sum --check --strict SHA256SUMS; \
    package_dir="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parent)')"; \
    test -f "$package_dir/srt/server_args.py"; \
    patch --batch --forward -p3 -d "$package_dir" < "$PATCH_FILE"; \
    python3 -c 'import pathlib, sglang, sys; assert pathlib.Path(sglang.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())' "$package_dir"; \
    python3 test_admin_auth_redaction.py --live; \
    install -m 0755 admin-key-entrypoint.sh /usr/local/bin/admin-key-entrypoint.sh
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/local/bin/admin-key-entrypoint.sh"]
