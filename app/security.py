"""
Anima - security utilities.

Provides:
  * Password hashing and verification using ``bcrypt``.
  * Symmetric field-level encryption/decryption using
    ``cryptography.fernet`` (keyed by the FERNET_KEY environment variable).

These helpers back the Users authentication supertype (password_hash) and the
encryption of sensitive PHI/PII fields before they are persisted to SQL Server.
"""

import os
import logging

import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("anima.security")

# =============================================================================
# Password hashing (bcrypt)
# =============================================================================
# bcrypt has a hard 72-byte input limit; longer passwords are silently
# truncated by the algorithm. We encode to UTF-8 explicitly.

def hash_password(password: str) -> str:
    """Hash a plaintext password with a per-password bcrypt salt.

    Returns the full modular-crypt hash string (algorithm + cost + salt +
    digest) suitable for storing in ``Users.password_hash``.
    """
    if not password:
        raise ValueError("Password must not be empty.")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Returns ``False`` on any malformed hash rather than raising, so auth
    endpoints can treat it uniformly as an invalid credential.
    """
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        logger.warning("Password verification received a malformed hash.")
        return False


# =============================================================================
# Symmetric encryption (cryptography.fernet)
# =============================================================================

def _load_fernet() -> Fernet:
    """Build the Fernet instance from the FERNET_KEY environment variable.

    The key must be a 32-byte url-safe base64-encoded string. Generate one with:
        python -c "from cryptography.fernet import Fernet; \\
                   print(Fernet.generate_key().decode())"
    """
    key = os.getenv("FERNET_KEY")
    if not key:
        raise RuntimeError(
            "FERNET_KEY is not set. Add it to your .env file. Generate with: "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid FERNET_KEY: {exc}") from exc


# Instantiated once at import time so a bad/missing key fails fast on startup.
_fernet = _load_fernet()


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value, returning a url-safe base64 token (str)."""
    if plaintext is None:
        raise ValueError("Cannot encrypt None.")
    token = _fernet.encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_value(token: str) -> str:
    """Decrypt a Fernet token back to its original string.

    Raises ``InvalidToken`` if the token is tampered with or was encrypted
    under a different key.
    """
    try:
        return _fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.error("Failed to decrypt value - invalid or tampered token.")
        raise


def generate_fernet_key() -> str:
    """Convenience helper to generate a fresh Fernet key (for setup scripts)."""
    return Fernet.generate_key().decode("utf-8")
