# ── Build stage ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install PDM
RUN pip install --no-cache-dir pdm

# Copy dependency files first for layer caching
COPY pyproject.toml pdm.lock ./

# Install production dependencies into a local __pypackages__ directory
# (no virtualenv, just a flat directory we can copy later)
RUN pdm install --prod --no-editable --no-self --no-isolation

# Copy source code and install the project itself
COPY src/ src/
COPY README.md LICENSE.md ./
RUN pdm install --prod --no-editable --no-isolation


# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="smithereens" \
      org.opencontainers.image.description="An experimental lightweight LLM interface" \
      org.opencontainers.image.source="https://github.com/acmelaw/smithereens" \
      org.opencontainers.image.licenses="MIT"

# Install runtime system dependencies (git + ripgrep used by tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 smithereens \
    && useradd --uid 1000 --gid smithereens --create-home smithereens

# Copy the installed virtualenv from the build stage
COPY --from=builder /build/.venv /app/.venv

# Put the venv on PATH
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create config/session directory
RUN mkdir -p /home/smithereens/.smithereens && \
    chown -R smithereens:smithereens /home/smithereens/.smithereens

WORKDIR /workspace
RUN chown smithereens:smithereens /workspace

USER smithereens

ENTRYPOINT ["smithereens"]
