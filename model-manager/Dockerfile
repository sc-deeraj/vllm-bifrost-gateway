FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 9000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=3)" || exit 1

# Single worker: the registry/locks are in-process memory, not safe to share
# across multiple workers/replicas without externalizing state.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
