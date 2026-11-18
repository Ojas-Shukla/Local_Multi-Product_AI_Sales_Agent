FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    SELLBIRD_HOME=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies and Chromium
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Create non-root application user
ARG SELLBIRD_UID=1000
ARG SELLBIRD_GID=1000

RUN test "$SELLBIRD_UID" -gt 0 && test "$SELLBIRD_GID" -gt 0 \
    && groupadd --gid "$SELLBIRD_GID" sellbird \
    && useradd --create-home --uid "$SELLBIRD_UID" --gid "$SELLBIRD_GID" sellbird \
    && mkdir -p /app/data \
    && chown -R sellbird:sellbird /app /ms-playwright

# Copy project files
COPY --chown=sellbird:sellbird sellbird/ sellbird/
COPY --chown=sellbird:sellbird prompts/ prompts/
COPY --chown=sellbird:sellbird skills/ skills/
COPY --chown=sellbird:sellbird sellbird.yaml .

USER sellbird

# Health check
HEALTHCHECK --interval=1m --timeout=10s --start-period=1m --retries=3 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 5555), timeout=5).close()"

ENTRYPOINT ["python", "-m", "sellbird"]
CMD ["dashboard"]
