"""
Non-invasive audit & lifecycle logger for Parkour <-> ALPR contract verification (B1-B5).
Active ONLY when environment variable ANPR_DEBUG_PARKOUR_LOG=1 is set.
Never logs secrets or unmasked license plates.
"""
import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import subprocess
import threading
import time
import uuid

_AUDIT_LOCK = threading.RLock()
_AUDIT_LOGGER = None
_AUDIT_LOG_PATH = None


def is_enabled():
    return os.environ.get("ANPR_DEBUG_PARKOUR_LOG", "0") == "1"


def now_iso_tz(epoch=None):
    dt = datetime.datetime.fromtimestamp(epoch if epoch is not None else time.time()).astimezone()
    return dt.isoformat(timespec="milliseconds")


def get_git_info(repo_dir=None):
    cwd = repo_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    commit_hash = "unknown"
    branch_name = "unknown"
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=2.0
        ).strip()
    except Exception:
        pass
    try:
        branch_name = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=2.0
        ).strip()
    except Exception:
        pass
    return commit_hash, branch_name


def mask_plate(plate_str):
    """Mask plate so real plates never appear in audit logs/reports: e.g. 'B 1786 CIC' -> 'B 17** *IC'."""
    if not plate_str:
        return ""
    s = str(plate_str).strip().upper()
    if s in ("NONE", "LIVE", "UNKNOWN", "TIDAK_TERBACA", "UNREADABLE", "NO_CAM"):
        return s
    m = re.match(r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{0,3})$', re.sub(r'[^A-Z0-9\s]', '', s))
    if m:
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        if len(digits) <= 2:
            d_mask = digits[0] + "*" * (len(digits) - 1)
        else:
            d_mask = digits[:2] + "*" * (len(digits) - 2)
        if not suffix:
            s_mask = ""
        elif len(suffix) == 1:
            s_mask = "*"
        else:
            s_mask = "*" * (len(suffix) - 2 if len(suffix) > 2 else 1) + suffix[-(2 if len(suffix) > 2 else 1):]
        return f"{prefix} {d_mask} {s_mask}".strip()
    clean = re.sub(r'[^A-Z0-9]', '', s)
    if len(clean) <= 3:
        return "*" * len(clean)
    return clean[:2] + ("*" * max(2, len(clean) - 4)) + clean[-2:]


def mask_path_plates(path_str):
    """Mask any license plate token embedded at the end of a capture filename."""
    if not path_str:
        return ""
    s = str(path_str)
    base = os.path.basename(s.replace("\\", "/"))
    dir_part = s[: len(s) - len(base)] if s.endswith(base) else ""
    m = re.match(r'^((?:ALPR_)?\d{8}_\d{6}_\d{3}_)([A-Za-z0-9]+)(\.[A-Za-z0-9]+)$', base)
    if m:
        prefix, token, ext = m.group(1), m.group(2), m.group(3)
        if token.upper() not in ("NONE", "LIVE", "NO_CAM", "UNKNOWN"):
            token = re.sub(r'\s+', '', mask_plate(token))
        return dir_part + f"{prefix}{token}{ext}"
    return s


def _get_logger():
    global _AUDIT_LOGGER, _AUDIT_LOG_PATH
    target_path = os.environ.get(
        "ANPR_PARKOUR_LOG_FILE",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "parkour_audit.jsonl"))
    )
    with _AUDIT_LOCK:
        if _AUDIT_LOGGER is not None and _AUDIT_LOG_PATH == target_path:
            return _AUDIT_LOGGER
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger = logging.getLogger("parkour_contract_audit")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        handler = RotatingFileHandler(target_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _AUDIT_LOGGER = logger
        _AUDIT_LOG_PATH = target_path
        return _AUDIT_LOGGER


def emit_json(record):
    if not is_enabled():
        return
    try:
        payload = dict(record)
        if "timestamp_iso" not in payload:
            payload["timestamp_iso"] = now_iso_tz()
        if "system_epoch_ms" not in payload:
            payload["system_epoch_ms"] = int(time.time() * 1000)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=False)
        _get_logger().info(line)
    except Exception:
        pass


def extract_osd_timestamp(frame):
    """
    If frame carries metadata or synthetic OSD timestamp attribute/dict (or env ANPR_SYNTHETIC_OSD_EPOCH),
    compute OSD timestamp & drift vs Pi/system clock (B5).
    """
    sys_now = time.time()
    osd_epoch = None
    osd_str = None
    env_osd = os.environ.get("ANPR_SYNTHETIC_OSD_EPOCH", "").strip()
    if env_osd:
        try:
            osd_epoch = float(env_osd)
            osd_str = datetime.datetime.fromtimestamp(osd_epoch).strftime("%d-%m-%Y %H:%M:%S")
        except Exception:
            pass
    return {
        "system_time_iso": now_iso_tz(sys_now),
        "system_epoch_ms": int(sys_now * 1000),
        "osd_time_str": osd_str,
        "osd_diff_sec": round(sys_now - osd_epoch, 3) if osd_epoch is not None else None,
    }


def new_request_id():
    return f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
