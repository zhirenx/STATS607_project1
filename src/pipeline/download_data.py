"""Download one SST-2 split from the Hugging Face Hub and verify its checksum.

The file is fetched from the dataset revision pinned in ``config.toml`` and
accepted only if its SHA-256 matches the recorded value, so every run of the
analysis sees byte-for-byte the data the original analysis used.

Usage::

    python -m src.pipeline.download_data --split validation
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi

from src.config import DEFAULT_CONFIG, load_config
from src.data import raw_path
from src.log import setup_logging

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """The data could not be obtained or did not match its checksum."""


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssl_context() -> ssl.SSLContext:
    """Return a TLS context using ``$SSL_CERT_FILE`` if set, else certifi's bundle.

    python.org builds of Python on macOS do not use the system certificate
    store, so plain ``urllib`` fails there without an explicit bundle.
    """
    cafile = os.environ.get("SSL_CERT_FILE") or certifi.where()
    try:
        return ssl.create_default_context(cafile=cafile)
    except (OSError, ssl.SSLError) as exc:
        raise DownloadError(
            f"cannot load TLS certificates from {cafile} ({exc}); "
            "unset SSL_CERT_FILE or point it to a PEM certificate bundle"
        ) from exc


def fetch(url: str, dest: Path, expected_sha256: str, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` unless a verified copy is already there.

    The download goes to a temporary file that replaces ``dest`` only after
    the checksum matches, so an interrupted or corrupted download never
    leaves a file that later steps would trust.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256sum(dest) == expected_sha256:
            log.info("%s already present and verified", dest)
            return dest
        log.warning("%s does not match its checksum; downloading it again", dest)

    partial = dest.with_name(dest.name + ".part")
    context = ssl_context()
    log.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
            with open(partial, "wb") as out:
                shutil.copyfileobj(response, out)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        if isinstance(getattr(exc, "reason", exc), ssl.SSLCertVerificationError):
            cause = ("the server's TLS certificate could not be verified; behind a proxy "
                     "that re-signs HTTPS traffic, set SSL_CERT_FILE to its certificate bundle")
        else:
            cause = f"{exc}; check the internet connection"
        raise DownloadError(
            f"could not download {url} ({cause}). Alternatively, download that URL in a "
            f"browser, save it as {dest} and rerun; it must have SHA-256 {expected_sha256}."
        ) from exc

    actual = sha256sum(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise DownloadError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}. "
            "The file on the server differs from the one the analysis used."
        )
    os.replace(partial, dest)
    log.info("saved %s (%d bytes, sha256 verified)", dest, dest.stat().st_size)
    return dest


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and download the requested split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, help="train or validation")
    parser.add_argument("--out", type=Path, help="destination (default: data/raw/sst2_<split>.parquet)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    setup_logging()

    config = load_config(args.config)
    if args.split not in config.data.splits:
        parser.error(f"--split must be one of {sorted(config.data.splits)}")
    split = config.data.splits[args.split]
    try:
        fetch(config.data.url(args.split), args.out or raw_path(args.split), split.sha256)
    except DownloadError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
