"""
Anima - FastAPI application entrypoint.

Initialises the FastAPI app, configures CORS middleware, mounts the static
directory that serves processed 2D MRI slices, and establishes the initial
SQL Server connection (via pyodbc) on startup.
"""

import os
import logging

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pyodbc

from database import get_db, wait_for_db, DB_NAME

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anima.main")

# --- Application -------------------------------------------------------------
app = FastAPI(
    title="Anima API",
    description="AI-integrated ADHD diagnosis support backend.",
    version="0.1.0",
)

# --- CORS middleware ---------------------------------------------------------
# PowerApps / Outlook clients call this API from different origins.
# Override with a comma-separated ALLOWED_ORIGINS env var in production.
_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if _origins_env.strip() == "*"
    else [o.strip() for o in _origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files: processed MRI slices --------------------------------------
# Matches the Docker volume mount at /app/static/mri_images.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(os.path.join(STATIC_DIR, "mri_images"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Lifecycle events --------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    """Block until SQL Server is reachable so the API doesn't serve traffic
    before its database dependency is available."""
    logger.info("Anima API starting up - checking SQL Server connectivity...")
    if not wait_for_db():
        # Log loudly but let the app start so /health can report the problem.
        logger.error("Startup completed WITHOUT a confirmed database connection.")
    else:
        logger.info("Database connectivity confirmed.")


# --- Routes ------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Basic liveness/info endpoint."""
    return {
        "service": "Anima API",
        "version": app.version,
        "status": "online",
    }


@app.get("/health")
def health(conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Readiness check that verifies a live query against SQL Server."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        cursor.fetchone()
        return {"status": "healthy", "database": DB_NAME}
    except pyodbc.Error as exc:
        logger.exception("Health check DB query failed.")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")