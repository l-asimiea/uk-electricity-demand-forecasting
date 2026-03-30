# The key principle here is a multi-stage build: a builder stage installs
# dependencies (slow, cached), and a runtime stage copies only what's needed (lean final image).

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (maximises layer cache)
COPY pyproject.toml uv.lock ./

# Install dependencies into an isolated location
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy the virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/
COPY dagster_home/ ./dagster_home/

# Put the venv on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV DAGSTER_HOME="/app/dagster_home"

# Dagster webserver port
EXPOSE 3000

CMD ["dagster", "dev", "-h", "0.0.0.0", "-p", "3000"]
