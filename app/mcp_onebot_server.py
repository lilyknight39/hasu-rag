"""
MCP server for sending audio files to a OneBot v11 HTTP server.
"""

from __future__ import annotations

import os
import base64
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import env_loader  # load .env if present

import requests
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("onebot-audio")

ONEBOT_BASE_URL = os.getenv("ONEBOT_BASE_URL", "http://127.0.0.1:5700").rstrip("/")
ONEBOT_ACCESS_TOKEN = os.getenv("ONEBOT_ACCESS_TOKEN", "")
ONEBOT_VOICE_DIR = Path(os.getenv("ONEBOT_VOICE_DIR", ".")).expanduser()
ONEBOT_VOICE_EXTS = os.getenv("ONEBOT_VOICE_EXTS", ".mp3")
ONEBOT_ALLOW_SUBDIRS = os.getenv("ONEBOT_ALLOW_SUBDIRS", "0") == "1"
ONEBOT_VOICE_RECURSIVE = os.getenv("ONEBOT_VOICE_RECURSIVE", "0") == "1"
ONEBOT_FILE_PREFIX = os.getenv("ONEBOT_FILE_PREFIX", "file://")
ONEBOT_RECORD_MODE = os.getenv("ONEBOT_RECORD_MODE", "path").strip().lower()
ONEBOT_BASE64_PREFIX = os.getenv("ONEBOT_BASE64_PREFIX", "base64://")
ONEBOT_PATH_MAP_FROM = os.getenv("ONEBOT_PATH_MAP_FROM", "").strip()
ONEBOT_PATH_MAP_TO = os.getenv("ONEBOT_PATH_MAP_TO", "").strip()
ONEBOT_TIMEOUT = float(os.getenv("ONEBOT_TIMEOUT", "10"))


def _parse_exts(raw_exts: str) -> list[str]:
    exts: list[str] = []
    for item in raw_exts.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        exts.append(item)
    return exts


VOICE_EXTS = _parse_exts(ONEBOT_VOICE_EXTS)
PATH_MAP_FROM = Path(ONEBOT_PATH_MAP_FROM).expanduser().resolve() if ONEBOT_PATH_MAP_FROM else None
PATH_MAP_TO = Path(ONEBOT_PATH_MAP_TO) if ONEBOT_PATH_MAP_TO else None


def _is_within_base(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _normalize_base_dir() -> Path:
    base = ONEBOT_VOICE_DIR.expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Voice base dir does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Voice base dir is not a directory: {base}")
    return base


def _sanitize_relative(name: str) -> Path:
    if not name or not name.strip():
        raise ValueError("file_name is required")
    rel = Path(name)
    if rel.is_absolute():
        raise ValueError("absolute paths are not allowed")
    if ".." in rel.parts:
        raise ValueError("parent traversal is not allowed")
    if len(rel.parts) > 1 and not ONEBOT_ALLOW_SUBDIRS:
        raise ValueError("subdirectories are not allowed")
    return rel


def _resolve_voice_file(file_name: str) -> Path:
    base = _normalize_base_dir()
    rel = _sanitize_relative(file_name)

    if rel.suffix:
        candidate = (base / rel).resolve()
        if _is_within_base(candidate, base) and candidate.exists():
            return candidate
    else:
        for ext in VOICE_EXTS:
            candidate = (base / f"{rel}{ext}").resolve()
            if _is_within_base(candidate, base) and candidate.exists():
                return candidate

    if ONEBOT_VOICE_RECURSIVE:
        search_root = base / rel.parent if rel.parent != Path(".") else base
        if search_root.exists():
            if rel.suffix:
                patterns = [rel.name]
            else:
                patterns = [f"{rel.name}{ext}" for ext in VOICE_EXTS] or [rel.name]
            for pattern in patterns:
                for match in search_root.rglob(pattern):
                    if match.is_file():
                        resolved = match.resolve()
                        if _is_within_base(resolved, base):
                            return resolved

    raise FileNotFoundError(f"Voice file not found under {base}: {file_name}")


def _map_path_for_onebot(file_path: Path) -> Path:
    if not PATH_MAP_FROM or not PATH_MAP_TO:
        return file_path
    resolved = file_path.resolve()
    try:
        rel = resolved.relative_to(PATH_MAP_FROM)
    except ValueError as exc:
        raise ValueError(f"Path map mismatch: {resolved} not under {PATH_MAP_FROM}") from exc
    return PATH_MAP_TO / rel


def _record_file_ref(file_path: Path) -> str:
    if ONEBOT_RECORD_MODE == "base64":
        data = base64.b64encode(file_path.read_bytes()).decode("ascii")
        return f"{ONEBOT_BASE64_PREFIX}{data}"
    mapped = _map_path_for_onebot(file_path)
    return f"{ONEBOT_FILE_PREFIX}{mapped}"


def _onebot_headers() -> Dict[str, str]:
    if not ONEBOT_ACCESS_TOKEN:
        return {}
    return {"Authorization": f"Bearer {ONEBOT_ACCESS_TOKEN}"}


def _onebot_post(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{ONEBOT_BASE_URL}/{action.lstrip('/')}"
    resp = requests.post(url, json=payload, headers=_onebot_headers(), timeout=ONEBOT_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"OneBot HTTP error {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError:
        return {"ok": resp.ok, "status_code": resp.status_code, "text": resp.text}


def _record_message(file_path: Path, caption: Optional[str]) -> list[Dict[str, Any]]:
    segments: list[Dict[str, Any]] = []
    if caption:
        segments.append({"type": "text", "data": {"text": caption}})
    file_value = _record_file_ref(file_path)
    segments.append({"type": "record", "data": {"file": file_value}})
    return segments


def _send_record(target_type: str, target_id: int, file_path: Path, caption: Optional[str]) -> Dict[str, Any]:
    message = _record_message(file_path, caption)
    if target_type == "group":
        payload = {"group_id": target_id, "message": message}
        return _onebot_post("send_group_msg", payload)
    if target_type == "private":
        payload = {"user_id": target_id, "message": message}
        return _onebot_post("send_private_msg", payload)
    raise ValueError("target_type must be 'group' or 'private'")


def _send_file(target_type: str, target_id: int, file_path: Path, display_name: Optional[str]) -> Dict[str, Any]:
    name = display_name or file_path.name
    mapped = _map_path_for_onebot(file_path)
    if target_type == "group":
        payload = {"group_id": target_id, "file": str(mapped), "name": name}
        return _onebot_post("upload_group_file", payload)
    if target_type == "private":
        payload = {"user_id": target_id, "file": str(mapped), "name": name}
        return _onebot_post("upload_private_file", payload)
    raise ValueError("target_type must be 'group' or 'private'")


@mcp.tool()
def send_voice(
    file_name: str,
    target_type: Literal["group", "private"],
    target_id: int,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a voice (record) message using a local audio file name.
    """
    file_path = _resolve_voice_file(file_name)
    result = _send_record(target_type, target_id, file_path, caption)
    return {"file_path": str(file_path), "onebot": result}


@mcp.tool()
def send_file(
    file_name: str,
    target_type: Literal["group", "private"],
    target_id: int,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload a file using a local audio file name.
    """
    file_path = _resolve_voice_file(file_name)
    result = _send_file(target_type, target_id, file_path, display_name)
    return {"file_path": str(file_path), "onebot": result}


if __name__ == "__main__":
    mcp.run()
