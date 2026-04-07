# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="CyberBattleEnv" \
      org.opencontainers.image.description="Multi-agent cybersecurity RL environment" \
      org.opencontainers.image.version="1.0.0"

# ── OS deps ───────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Install the package so imports resolve cleanly
RUN pip install --no-cache-dir .

# ── Runtime config ────────────────────────────────────────────────────────────
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    MAX_CONCURRENT_ENVS=100 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

# Expose server port
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["sh", "-c", \
     "uvicorn cyber_battle_env.server.app:app \
      --host $HOST \
      --port $PORT \
      --workers $WORKERS \
      --log-level $LOG_LEVEL"]
