"""MongoDB layer: accounts, daily credits and video history.

Metadata only. Videos stay on the filesystem where the existing pipeline already
puts them; Mongo holds the row that points at them.

Configuration comes from the environment (see .env.example). Nothing here has a
hardcoded credential or secret.

The credit rules live in this module rather than in the API layer, because they
have to be atomic to be worth anything. A read-then-write in the request handler
would let two near-simultaneous uploads both pass a "credits >= 5" check and both
be charged from the same balance. Every mutation below is a single MongoDB
command whose filter contains the condition it depends on.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("trafficintel.db")

ROOT = Path(__file__).resolve().parent.parent

DAILY_CREDITS = int(os.environ.get("DAILY_CREDITS", "15"))
CREDITS_PER_VIDEO = int(os.environ.get("CREDITS_PER_VIDEO", "5"))

# 15 / 5 = 3 videos per day. Derived rather than configured separately, so the
# number shown in the UI can never disagree with what the backend enforces.
VIDEOS_PER_DAY = DAILY_CREDITS // CREDITS_PER_VIDEO


def _load_env_file(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ.

    Hand-rolled instead of pulling in python-dotenv: it is a dozen lines, and one
    fewer dependency is one fewer thing that can fail to install the day before a
    deadline. Existing environment variables always win, so a real shell export
    overrides the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "trafficintel")

_client = None
_db = None
_init_error: Optional[str] = None


class DatabaseUnavailable(RuntimeError):
    """Mongo could not be reached. Raised so the API can answer 503, not 500."""


def _today() -> str:
    """The reset key, as YYYY-MM-DD in the server's local timezone.

    A date string rather than a timestamp because the rule is "once per day", and
    comparing a stored day to today's day is exact. Subtracting 24 hours from a
    timestamp would let a user who ran at 23:00 get fresh credits at 23:01 the
    next day but not at 09:00, which is not what "daily" means to anyone.
    """
    return datetime.now().strftime("%Y-%m-%d")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def connect(timeout_ms: int = 3000):
    """Open the connection and create indexes. Safe to call repeatedly."""
    global _client, _db, _init_error
    if _db is not None:
        return _db
    try:
        from pymongo import ASCENDING, MongoClient
    except ImportError as exc:                                        # noqa: BLE001
        _init_error = ("pymongo is not installed. Run: "
                       "pip install -r requirements.txt")
        raise DatabaseUnavailable(_init_error) from exc

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=timeout_ms,
                             tz_aware=True)
        # MongoClient is lazy, so without this the first real failure would
        # surface inside an unrelated request handler minutes later.
        client.admin.command("ping")
        database = client[MONGODB_DATABASE]

        database.users.create_index([("email", ASCENDING)], unique=True)
        database.video_history.create_index([("user_id", ASCENDING),
                                            ("created_at", ASCENDING)])
        # request_id is the _id of credit_charges, so uniqueness - and therefore
        # protection against a double-submitted upload being charged twice - comes
        # from the primary key and needs no extra index.
    except Exception as exc:                                          # noqa: BLE001
        _init_error = f"{type(exc).__name__}: {exc}"
        log.error("MongoDB unavailable at %s (%s)", MONGODB_URI, _init_error)
        raise DatabaseUnavailable(
            f"Could not connect to MongoDB at {MONGODB_URI}. "
            f"Is mongod running? ({_init_error})"
        ) from exc

    _client, _db = client, database
    _init_error = None
    log.info("MongoDB connected | database=%s", MONGODB_DATABASE)
    return _db


def get_db():
    if _db is not None:
        return _db
    return connect()


def status() -> dict:
    """Connection state for /api/health. Never raises."""
    try:
        get_db()
        return {"connected": True, "database": MONGODB_DATABASE}
    except DatabaseUnavailable as exc:
        return {"connected": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #

def create_user(name: str, email: str, password_hash: str) -> dict:
    """Insert a new account. Raises ValueError if the email is taken."""
    from pymongo.errors import DuplicateKeyError

    doc = {
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "daily_credits": DAILY_CREDITS,
        "last_credit_reset": _today(),
        "created_at": _now(),
    }
    try:
        result = get_db().users.insert_one(doc)
    except DuplicateKeyError as exc:
        raise ValueError("An account with that email already exists.") from exc
    doc["_id"] = result.inserted_id
    return doc


def find_by_email(email: str) -> Optional[dict]:
    return get_db().users.find_one({"email": email})


def get_user(user_id) -> Optional[dict]:
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(str(user_id))
    except (InvalidId, TypeError):
        return None
    return get_db().users.find_one({"_id": oid})


# --------------------------------------------------------------------------- #
# Credits
# --------------------------------------------------------------------------- #

def apply_daily_reset(user_id) -> dict:
    """Top the balance back up if the stored reset date is not today.

    This is the whole daily-reset mechanism: it runs on the user's next request
    after midnight, so there is no scheduler, no background thread and nothing to
    keep alive between runs. The filter carries the condition, so two concurrent
    requests cannot both reset and hand out 30 credits.
    """
    from bson import ObjectId
    from pymongo import ReturnDocument

    oid = ObjectId(str(user_id))
    today = _today()
    updated = get_db().users.find_one_and_update(
        {"_id": oid, "last_credit_reset": {"$ne": today}},
        {"$set": {"daily_credits": DAILY_CREDITS, "last_credit_reset": today}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is not None:
        log.info("Credits reset to %d for user %s (new day: %s)",
                 DAILY_CREDITS, user_id, today)
        return updated
    return get_db().users.find_one({"_id": oid})


def credit_state(user_id) -> dict:
    """What the UI shows. Applies the reset first so the number is current."""
    user = apply_daily_reset(user_id)
    remaining = int(user.get("daily_credits", 0)) if user else 0
    used = max(0, DAILY_CREDITS - remaining)
    return {
        "credits_remaining": remaining,
        "credits_total": DAILY_CREDITS,
        "credits_per_video": CREDITS_PER_VIDEO,
        "videos_today": used // CREDITS_PER_VIDEO,
        "videos_per_day": VIDEOS_PER_DAY,
        "can_process": remaining >= CREDITS_PER_VIDEO,
    }


def reserve_credits(user_id, request_id: str) -> dict:
    """Charge one video's worth of credits, atomically and at most once.

    Returns {"ok": True, "remaining": n} or {"ok": False, "reason": ..., ...}.

    Two separate protections, because they guard different failures:

    1. A marker document keyed on request_id is inserted first. Its _id is the
       request id, so a duplicate submit - a double-clicked button, a retried
       XHR - collides on the primary key and is refused instead of being charged
       a second time.
    2. The deduction itself is a find_one_and_update whose filter requires
       daily_credits >= cost. MongoDB applies that filter and the decrement as
       one operation, so the balance cannot go negative no matter how many
       requests arrive at once.

    If the balance turns out to be too low, the marker is removed again: the
    request was refused, so its id should not be permanently burned.
    """
    from bson import ObjectId
    from pymongo import ReturnDocument
    from pymongo.errors import DuplicateKeyError

    database = get_db()
    oid = ObjectId(str(user_id))

    try:
        database.credit_charges.insert_one({
            "_id": request_id,
            "user_id": oid,
            "amount": CREDITS_PER_VIDEO,
            "created_at": _now(),
        })
    except DuplicateKeyError:
        state = credit_state(user_id)
        return {"ok": False, "reason": "duplicate",
                "remaining": state["credits_remaining"]}

    apply_daily_reset(oid)

    updated = database.users.find_one_and_update(
        {"_id": oid, "daily_credits": {"$gte": CREDITS_PER_VIDEO}},
        {"$inc": {"daily_credits": -CREDITS_PER_VIDEO}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        database.credit_charges.delete_one({"_id": request_id})
        user = database.users.find_one({"_id": oid}) or {}
        return {"ok": False, "reason": "insufficient",
                "remaining": int(user.get("daily_credits", 0))}

    remaining = int(updated["daily_credits"])
    log.info("Charged %d credits to user %s | %d remaining",
             CREDITS_PER_VIDEO, user_id, remaining)
    return {"ok": True, "remaining": remaining}


def refund_credits(user_id, request_id: str) -> Optional[int]:
    """Give the credits back when processing failed through no fault of the user.

    Deleting the charge marker first makes this idempotent: if a refund somehow
    runs twice, the second delete matches nothing and no credits are returned a
    second time. Capped at the daily total so a refund can never inflate a
    balance above 15.
    """
    from bson import ObjectId

    database = get_db()
    oid = ObjectId(str(user_id))
    if database.credit_charges.delete_one({"_id": request_id}).deleted_count == 0:
        return None

    user = database.users.find_one({"_id": oid}) or {}
    restored = min(DAILY_CREDITS,
                   int(user.get("daily_credits", 0)) + CREDITS_PER_VIDEO)
    database.users.update_one({"_id": oid},
                              {"$set": {"daily_credits": restored}})
    log.info("Refunded %d credits to user %s | %d available",
             CREDITS_PER_VIDEO, user_id, restored)
    return restored


# --------------------------------------------------------------------------- #
# Video history
# --------------------------------------------------------------------------- #

def record_job(job_id: str, user_id, request_id: str, original_filename: str,
               output_filename: str, options: dict) -> None:
    """One row per accepted job. Metadata only - no video bytes."""
    from bson import ObjectId

    get_db().video_history.insert_one({
        "_id": job_id,
        "user_id": ObjectId(str(user_id)),
        "request_id": request_id,
        "original_filename": original_filename,
        "output_filename": output_filename,
        "status": "processing",
        "credits_used": CREDITS_PER_VIDEO,
        "options": options,
        "created_at": _now(),
        "finished_at": None,
    })


def finish_job(job_id: str, status: str, error: Optional[str] = None,
               summary: Optional[dict] = None, credits_used: Optional[int] = None) -> None:
    fields = {"status": status, "finished_at": _now(), "error": error}
    if summary is not None:
        fields["summary"] = summary
    if credits_used is not None:
        fields["credits_used"] = credits_used
    get_db().video_history.update_one({"_id": job_id}, {"$set": fields})


def job_owner(job_id: str) -> Optional[str]:
    """Which user owns this job, for the ownership check on result endpoints.

    Read from Mongo rather than kept on the in-memory Job so that ownership
    survives a server restart, and so jobs.py does not have to learn about users.
    """
    doc = get_db().video_history.find_one({"_id": job_id}, {"user_id": 1})
    return str(doc["user_id"]) if doc else None


def history_for(user_id, limit: int = 100) -> list:
    from bson import ObjectId

    rows = get_db().video_history.find(
        {"user_id": ObjectId(str(user_id))}
    ).sort("created_at", -1).limit(limit)

    out = []
    for row in rows:
        created = row.get("created_at")
        out.append({
            "job_id": row["_id"],
            "original_filename": row.get("original_filename"),
            "status": row.get("status"),
            "credits_used": row.get("credits_used", 0),
            "created_at": created.isoformat() if created else None,
            "error": row.get("error"),
            "summary": row.get("summary"),
            # Whether the file is still on disk is answered by the API layer,
            # which owns the paths; Mongo only says a result was produced.
            "has_output": row.get("status") == "completed",
        })
    return out


def public_user(user: dict) -> dict:
    """Strip the hash before anything about a user is sent to a client."""
    return {
        "id": str(user["_id"]),
        "name": user.get("name"),
        "email": user.get("email"),
        "created_at": (user["created_at"].isoformat()
                       if isinstance(user.get("created_at"), datetime) else None),
    }
