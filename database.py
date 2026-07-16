"""
Anima - database connection layer.

Establishes connections to Microsoft SQL Server via pyodbc using credentials
loaded from the environment (populated from the local .env file / Docker).
"""

import os
import time
import logging

import pyodbc
from dotenv import load_dotenv

# Load .env when running outside Docker (Docker injects vars directly).
load_dotenv()

logger = logging.getLogger("anima.database")

# --- Connection settings (from environment) ----------------------------------
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "1433")
DB_NAME = os.getenv("DB_NAME", "anima")
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
ODBC_DRIVER = os.getenv("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")


def build_connection_string(database: str | None = None) -> str:
    """Assemble a pyodbc connection string for SQL Server.

    ``TrustServerCertificate=yes`` is used because the containerised SQL Server
    presents a self-signed certificate in local/dev environments.
    """
    target_db = database if database is not None else DB_NAME
    return (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={DB_HOST},{DB_PORT};"
        f"DATABASE={target_db};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )


def get_connection(database: str | None = None) -> pyodbc.Connection:
    """Return a new pyodbc connection.

    Callers are responsible for closing the connection (or using it as a
    context manager). Use :func:`get_db` for a FastAPI dependency that handles
    cleanup automatically.
    """
    return pyodbc.connect(build_connection_string(database), autocommit=False)


def get_db():
    """FastAPI dependency that yields a connection and guarantees close.

    Usage::

        @app.get("/patients")
        def list_patients(conn = Depends(get_db)):
            ...
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def wait_for_db(max_retries: int = 15, delay_seconds: int = 4) -> bool:
    """Poll SQL Server until it accepts connections.

    Even with docker-compose ``depends_on: service_healthy``, this adds a
    resilient startup check. Connects to the ``master`` database because the
    application database may not exist yet on first boot.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            conn = pyodbc.connect(build_connection_string(database="master"), timeout=5)
            conn.close()
            logger.info("SQL Server is available (attempt %s).", attempt)
            return True
        except pyodbc.Error as exc:  # noqa: PERF203
            last_error = exc
            logger.warning(
                "SQL Server not ready (attempt %s/%s): %s",
                attempt, max_retries, exc,
            )
            time.sleep(delay_seconds)

    logger.error("Could not connect to SQL Server after %s attempts: %s",
                 max_retries, last_error)
    return False