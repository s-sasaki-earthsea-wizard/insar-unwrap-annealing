# syntax=docker/dockerfile:1
# CPU-only image for the insar-unwrap skeleton.
# Pinned versions verified on 2026-11-02; bump deliberately, not via :latest.
FROM python:3.12.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ARG UID=1000
ARG GID=1000

# The project environment lives outside the bind-mounted source tree so a host
# .venv can never shadow it. A persistent cache dir lets `uv run` see that the
# editable project is already fresh instead of rebuilding it on every call.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/opt/uv-cache \
    PATH="/opt/venv/bin:${PATH}"

RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash app \
    && mkdir -p /opt/venv /opt/uv-cache /work \
    && chown -R app:app /opt/venv /opt/uv-cache /work

WORKDIR /work
USER app

# Dependency layer: only the lockfile and project metadata.
COPY --chown=app:app pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --group dev --no-install-project

# Project layer: source and tests, then install the project itself.
COPY --chown=app:app src ./src
COPY --chown=app:app tests ./tests
COPY --chown=app:app configs ./configs
RUN uv sync --locked --group dev

# The lockfile is enforced at build time (uv sync --locked). At run time the
# image is immutable, so commands execute straight from /opt/venv/bin: no
# re-sync, and no spurious rebuild of the editable project.
CMD ["insar-unwrap", "--help"]
