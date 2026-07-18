# -----------------------------------------------------------------------------
# Anima FastAPI backend image
# Includes the Microsoft ODBC Driver 18 so pyodbc can reach SQL Server, and bakes
# Bio_ClinicalBERT into the image so the API runs fully offline at runtime.
# -----------------------------------------------------------------------------
# Pinned to bookworm (Debian 12) so it matches Microsoft's debian/12 ODBC repo.
# (Plain python:3.11-slim now tracks Debian 13 "trixie", for which that repo has
# no packages.)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- System deps + Microsoft ODBC Driver 18 for SQL Server -------------------
# The apt source line is written explicitly (single option bracket) instead of
# downloading Microsoft's prod.list and sed-injecting signed-by, which produced
# a malformed double-bracket entry with newer prod.list files.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl gnupg2 apt-transport-https ca-certificates \
        unixodbc-dev gcc g++ \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps -------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Bake the clinical language model into the image -------------------------
# Download Bio_ClinicalBERT into the image's Hugging Face cache at build time, so
# the API needs NO network at runtime. HF_HOME sets the cache root for both the
# build-time download and the runtime load. Placed before the app COPY so editing
# application code doesn't invalidate this (expensive) layer.
ENV HF_HOME=/opt/hf-cache
COPY scripts/predownload_model.py /tmp/predownload_model.py
RUN python /tmp/predownload_model.py && rm -f /tmp/predownload_model.py

# From here on, load the model from the baked cache ONLY (no network at runtime).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- Application source ------------------------------------------------------
COPY ./app /app

# Ensure the MRI image output directory exists
RUN mkdir -p /app/static/mri_images

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
