# ============================================================
# Stage 1 — dependency installer (uv)
# ============================================================
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install all production deps into /app/.venv
# --no-install-project: don't install the project itself yet
RUN uv sync --frozen --no-install-project --no-dev

# ============================================================
# Stage 2 — runtime image
# ============================================================
FROM python:3.13-slim AS runtime

# System packages needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY . .

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Disable tokenizers parallelism warning inside containers
    TOKENIZERS_PARALLELISM=false

# Non-root user — trading app doesn't need root
RUN useradd -m -u 1001 trader && chown -R trader:trader /app
USER trader

# Default: run the orchestrator (overridden in compose for other services)
CMD ["python", "-m", "orchestrator.main"]
