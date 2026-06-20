FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POWERNZ_WEB_DATA_DIR=/data \
    POWERNZ_WEB_HOST=0.0.0.0 \
    POWERNZ_SECURE_COOKIES=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg gosu libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . ./

# The model files are checked against their hashes by the existing downloader.
# They are baked into the private image, never fetched for each visitor upload.
RUN python model_downloader.py

RUN useradd --create-home --uid 10001 powernz \
    && mkdir -p /data \
    && chown -R powernz:powernz /app /data

COPY deploy/docker-entrypoint.sh /usr/local/bin/powernz-entrypoint
RUN chmod 755 /usr/local/bin/powernz-entrypoint

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

ENTRYPOINT ["/usr/local/bin/powernz-entrypoint"]
CMD ["python", "-m", "web"]
