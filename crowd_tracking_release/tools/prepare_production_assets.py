#!/usr/bin/env python3
"""Verify and explicitly provision production model checkpoints.

The repository stores the small asset manifest, not model binaries. This tool
never downloads a model. Its default action is verification; passing
``--source-dir`` may copy only missing assets from an already trusted local
directory with the same relative layout. Replacing an existing asset requires
the explicit ``--overwrite`` opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "production-assets.json"
CHUNK_SIZE = 1024 * 1024


class ManifestError(ValueError):
    """Raised when the checked-in asset manifest is malformed."""


@dataclass(frozen=True)
class Asset:
    """One immutable runtime asset declared by the production manifest."""

    identifier: str
    relative_path: Path
    sha256: str
    size_bytes: int

    @property
    def display_path(self) -> str:
        return self.relative_path.as_posix()


def _relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty POSIX relative path.")
    if "\\" in value:
        raise ManifestError(f"{field} must use '/' separators.")

    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"{field} must stay below the project root.")
    return Path(*path.parts)


def _sha256_value(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ManifestError(f"{field} must be a 64-character SHA-256 hex string.")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ManifestError(f"{field} must be a SHA-256 hex string.")
    return normalized


def load_manifest(path: Path) -> list[Asset]:
    """Load the versioned manifest and reject unsafe or ambiguous entries."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"Manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"Invalid JSON in manifest {path}: {error.msg}") from error

    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ManifestError("Manifest schema_version must be 1.")
    raw_assets = raw.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ManifestError("Manifest assets must be a non-empty list.")

    assets: list[Asset] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, item in enumerate(raw_assets):
        prefix = f"assets[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{prefix} must be an object.")

        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ManifestError(f"{prefix}.id must be a non-empty string.")
        if identifier in seen_ids:
            raise ManifestError(f"Duplicate asset id: {identifier}")

        relative_path = _relative_path(item.get("path"), field=f"{prefix}.path")
        if relative_path in seen_paths:
            raise ManifestError(f"Duplicate asset path: {relative_path.as_posix()}")

        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            raise ManifestError(f"{prefix}.size_bytes must be a positive integer.")

        asset = Asset(
            identifier=identifier,
            relative_path=relative_path,
            sha256=_sha256_value(item.get("sha256"), field=f"{prefix}.sha256"),
            size_bytes=size_bytes,
        )
        assets.append(asset)
        seen_ids.add(identifier)
        seen_paths.add(relative_path)
    return assets


def _asset_path(root: Path, asset: Asset) -> Path:
    """Resolve an already-sanitized manifest path defensively under ``root``."""

    root = root.resolve()
    candidate = (root / asset.relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:  # Defensive in case a future manifest parser changes.
        raise ManifestError(f"Asset path escapes the project root: {asset.display_path}") from error
    return candidate


def sha256_file(path: Path) -> str:
    """Return a streamed SHA-256 digest without loading a checkpoint into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_status(path: Path, asset: Asset) -> tuple[str, str]:
    """Classify ``path`` as valid, missing, or invalid for an asset declaration."""

    if not path.exists():
        return "missing", "file is absent"
    if not path.is_file():
        return "invalid", "path is not a regular file"
    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        return "invalid", f"size {actual_size} bytes does not match expected {asset.size_bytes}"
    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        return "invalid", f"SHA-256 {actual_sha256} does not match manifest"
    return "valid", "verified"


def _copy_stream(source: Path, destination: Path) -> None:
    """Copy a file while writing to a caller-selected safe destination."""

    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=CHUNK_SIZE)


def copy_verified_asset(source: Path, destination: Path, asset: Asset, *, overwrite: bool) -> tuple[bool, str]:
    """Copy a verified source asset, never replacing a target without opt-in.

    A copied file is hashed again before success is reported. Normal copies use
    exclusive creation (``xb``), so a concurrently created destination is not
    overwritten either.
    """

    source_state, source_detail = asset_status(source, asset)
    if source_state != "valid":
        return False, f"source is {source_state}: {source_detail}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    created_destination = False
    try:
        if overwrite:
            temporary_path = destination.with_name(
                f".{destination.name}.prepare-{uuid.uuid4().hex}.part"
            )
            _copy_stream(source, temporary_path)
            copied_state, copied_detail = asset_status(temporary_path, asset)
            if copied_state != "valid":
                return False, f"copied file is {copied_state}: {copied_detail}"
            os.replace(temporary_path, destination)
            temporary_path = None
        else:
            # Mark ownership only after exclusive creation succeeds. This lets
            # the finally block remove a partial file on a copy error without
            # ever unlinking a concurrently-created destination.
            with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
                created_destination = True
                shutil.copyfileobj(source_handle, destination_handle, length=CHUNK_SIZE)
            copied_state, copied_detail = asset_status(destination, asset)
            if copied_state != "valid":
                return False, f"copied file is {copied_state}: {copied_detail}"
        return True, "copied and checksum verified"
    except FileExistsError:
        return False, "destination already exists; rerun with --overwrite to replace it"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if created_destination:
            final_state, _ = asset_status(destination, asset)
            if final_state != "valid":
                destination.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify production model assets. With --source-dir, copy only missing "
            "assets from that local directory; this tool never downloads models."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Trusted local directory containing the manifest's relative asset paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of existing or invalid destination assets (requires --source-dir).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Asset manifest to use (default: {DEFAULT_MANIFEST.relative_to(PROJECT_ROOT).as_posix()}).",
    )
    return parser


def prepare_assets(
    assets: Iterable[Asset],
    *,
    project_root: Path,
    source_dir: Path | None,
    overwrite: bool,
) -> int:
    """Verify all assets and optionally provision missing files from ``source_dir``."""

    failures = 0
    for asset in assets:
        destination = _asset_path(project_root, asset)
        destination_state, destination_detail = asset_status(destination, asset)
        if destination_state == "valid" and not overwrite:
            print(f"[verified] {asset.display_path}")
            continue

        if source_dir is None:
            print(f"[{destination_state}] {asset.display_path}: {destination_detail}", file=sys.stderr)
            failures += 1
            continue

        if destination_state != "missing" and not overwrite:
            print(
                f"[{destination_state}] {asset.display_path}: {destination_detail}. "
                "Refusing to overwrite; rerun with --overwrite only if the source is trusted.",
                file=sys.stderr,
            )
            failures += 1
            continue

        source = _asset_path(source_dir, asset)
        copied, detail = copy_verified_asset(source, destination, asset, overwrite=overwrite)
        if copied:
            print(f"[copied] {asset.display_path}: {detail}")
        else:
            print(f"[failed] {asset.display_path}: {detail}", file=sys.stderr)
            failures += 1

    if failures:
        print(
            f"Asset preparation failed for {failures} item(s). See docs/model-assets.md for the local provisioning flow.",
            file=sys.stderr,
        )
        return 1
    print("All production model assets are verified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.overwrite and arguments.source_dir is None:
        parser.error("--overwrite requires --source-dir; this tool never downloads assets.")

    project_root = PROJECT_ROOT.resolve()
    manifest_path = arguments.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    try:
        assets = load_manifest(manifest_path.resolve())
    except ManifestError as error:
        print(f"[manifest-error] {error}", file=sys.stderr)
        return 2

    source_dir = arguments.source_dir
    if source_dir is not None:
        source_dir = source_dir.resolve()
        if not source_dir.is_dir():
            print(f"[source-error] Source directory does not exist: {source_dir}", file=sys.stderr)
            return 2

    return prepare_assets(
        assets,
        project_root=project_root,
        source_dir=source_dir,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
