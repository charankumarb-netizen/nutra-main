"""
Privacy-First Personal Health Vault (local, file based).

  * nothing is stored without explicit consent;
  * AI-generated summaries (masked) and original files live in SEPARATE folders; the original is kept only on request;
  * the owner can list, export, and delete one item or everything; items expire automatically (retention);
  * the owner is identified by a secret token (>= 16 characters) that is only stored as a hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .privacy import PRIVACY_NOTICE, mask_structure
from .timeline import build_timeline

DEFAULT_RETENTION_DAYS = 90
_ID = re.compile(r"^[0-9a-f]{32}$")


class VaultError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HealthVault:
    def __init__(self, root: str | Path, retention_days: int | None = None) -> None:
        self.root = Path(root)
        self.retention_days = int(retention_days or os.getenv("NUTRICHECK_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))

    # -- paths --------------------------------------------------------------
    def _owner_dir(self, token: str) -> Path:
        if not isinstance(token, str) or len(token) < 16:
            raise VaultError("The vault key must be a secret of at least 16 characters.", 400)
        return self.root / hashlib.sha256(token.encode()).hexdigest()[:32]

    def _paths(self, token: str) -> tuple[Path, Path]:
        base = self._owner_dir(token)
        return base / "summaries", base / "originals"

    # -- operations -----------------------------------------------------------
    def store(self, token: str, analysis: dict, *, consent: bool, label: str = "", original: bytes | None = None,
              store_original: bool = False, filename: str = "", now: datetime | None = None) -> dict:
        if consent is not True:
            raise VaultError("Storing a report requires explicit permission (consent=true). Nothing was stored.", 403)
        summaries, originals = self._paths(token)
        summaries.mkdir(parents=True, exist_ok=True)
        now = now or _now()
        rid = uuid.uuid4().hex
        record = {
            "id": rid, "label": mask_structure(label)[:120], "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=self.retention_days)).isoformat(),
            "original_stored": False, "sha256": hashlib.sha256(original).hexdigest() if original is not None else None,
            "analysis": mask_structure(analysis),
            "note": "AI-generated summary; identifiers masked by pattern matching. The original report is not part of this record.",
        }
        if store_original:
            if original is None:
                raise VaultError("store_original was requested but no original file was provided.", 400)
            originals.mkdir(parents=True, exist_ok=True)
            (originals / f"{rid}.bin").write_bytes(original)
            record["original_stored"] = True
            record["original_filename"] = Path(filename).name[:120]
        (summaries / f"{rid}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return {k: record[k] for k in ("id", "label", "created_at", "expires_at", "original_stored")}

    def _load(self, token: str, rid: str) -> dict:
        if not _ID.match(rid or ""):
            raise VaultError("Invalid report id.", 400)
        p = self._paths(token)[0] / f"{rid}.json"
        if not p.exists():
            raise VaultError("Report not found.", 404)
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self, token: str) -> list[dict]:
        summaries, _ = self._paths(token)
        if not summaries.exists():
            return []
        out = []
        for f in sorted(summaries.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            out.append({k: r[k] for k in ("id", "label", "created_at", "expires_at", "original_stored")})
        return sorted(out, key=lambda r: r["created_at"])

    def get(self, token: str, rid: str) -> dict:
        return self._load(token, rid)

    def get_original(self, token: str, rid: str) -> bytes:
        rec = self._load(token, rid)
        if not rec["original_stored"]:
            raise VaultError("The original file was not stored (it is only kept on request).", 404)
        return (self._paths(token)[1] / f"{rid}.bin").read_bytes()

    def delete(self, token: str, rid: str) -> dict:
        self._load(token, rid)
        summaries, originals = self._paths(token)
        (summaries / f"{rid}.json").unlink(missing_ok=True)
        (originals / f"{rid}.bin").unlink(missing_ok=True)
        return {"deleted": rid}

    def delete_all(self, token: str) -> dict:
        base = self._owner_dir(token)
        n = len(self.list(token))
        if base.exists():
            shutil.rmtree(base)
        return {"deleted_reports": n}

    def purge_expired(self, now: datetime | None = None) -> int:
        """Delete every expired item of every owner (call periodically)."""
        now = now or _now()
        removed = 0
        if not self.root.exists():
            return 0
        for owner in self.root.iterdir():
            for f in (owner / "summaries").glob("*.json") if (owner / "summaries").exists() else []:
                r = json.loads(f.read_text(encoding="utf-8"))
                if datetime.fromisoformat(r["expires_at"]) <= now:
                    f.unlink(missing_ok=True)
                    (owner / "originals" / f"{f.stem}.bin").unlink(missing_ok=True)
                    removed += 1
        return removed

    def timeline(self, token: str) -> dict:
        analyses = []
        for item in self.list(token):
            rec = self._load(token, item["id"])
            a = rec["analysis"]
            a["report_id"] = rec["label"] or rec["id"][:8]
            analyses.append(a)
        return build_timeline(analyses)

    def export(self, token: str) -> dict:
        """Everything the owner has stored, as one JSON document (data portability)."""
        return {"exported_at": _now().isoformat(), "reports": [self._load(token, i["id"]) for i in self.list(token)],
                "timeline": self.timeline(token), "privacy": PRIVACY_NOTICE}

    @staticmethod
    def notice(retention_days: int | None = None) -> dict:
        n = dict(PRIVACY_NOTICE)
        if retention_days:
            n["retention"] = n["retention"].replace("default 90 days, set by the operator", f"currently {retention_days} days")
        return n
