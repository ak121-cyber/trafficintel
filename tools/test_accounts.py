"""Unit tests for accounts, sessions and the daily credit system.

Runs without MongoDB, fastapi, torch or a GPU:

    python tools/test_accounts.py

MongoDB is replaced by a small in-memory stand-in (below) that enforces the two
behaviours the credit logic actually leans on: unique keys, and a document update
whose filter is evaluated atomically with its modification. Everything else -
backend/db.py and backend/auth.py - is the real code, imported unmodified.

What these tests are for: the credit rules are a server-side spending limit, so
the properties worth proving are that the balance can never go negative, that one
upload cannot be charged twice, that a refused request costs nothing, and that a
password is never stored or returned in a readable form. Those are all things a
future edit could quietly break.
"""

from __future__ import annotations

import ast
import copy
import os
import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fixed configuration, set before backend.db is imported: it reads these once at
# import time. Pinning them here means the suite proves the documented 15/5/3
# rule rather than whatever happens to be in the local .env.
os.environ["DAILY_CREDITS"] = "15"
os.environ["CREDITS_PER_VIDEO"] = "5"
# Supplying a secret keeps backend/auth.py from generating one and writing
# .jwt_secret into the checkout as a side effect of running the tests.
os.environ["JWT_SECRET"] = "test-only-secret-not-used-anywhere-real"


# --------------------------------------------------------------------------- #
# In-memory MongoDB stand-in
# --------------------------------------------------------------------------- #

class InvalidId(Exception):
    pass


class DuplicateKeyError(Exception):
    pass


class ObjectId:
    """Value-typed 12-byte id, enough of bson.ObjectId for db.py."""

    _counter = 0
    _lock = threading.Lock()

    def __init__(self, value=None):
        if value is None:
            with ObjectId._lock:
                ObjectId._counter += 1
                self._hex = f"{ObjectId._counter:024x}"
            return
        if isinstance(value, ObjectId):
            self._hex = value._hex
            return
        text = str(value)
        if len(text) != 24:
            raise InvalidId(f"{text!r} is not a valid ObjectId")
        try:
            int(text, 16)
        except ValueError as exc:
            raise InvalidId(f"{text!r} is not a valid ObjectId") from exc
        self._hex = text

    def __str__(self):
        return self._hex

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, ObjectId) and other._hex == self._hex

    def __hash__(self):
        return hash(self._hex)


def _matches(doc: dict, query: dict) -> bool:
    for field, condition in query.items():
        value = doc.get(field)
        if isinstance(condition, dict):
            for op, operand in condition.items():
                if op == "$ne" and value == operand:
                    return False
                if op == "$gte" and not (value is not None and value >= operand):
                    return False
                if op not in ("$ne", "$gte"):
                    raise NotImplementedError(f"fake mongo: operator {op}")
        elif value != condition:
            return False
    return True


def _apply(doc: dict, update: dict) -> None:
    for op, fields in update.items():
        if op == "$set":
            doc.update(fields)
        elif op == "$inc":
            for field, delta in fields.items():
                doc[field] = doc.get(field, 0) + delta
        else:
            raise NotImplementedError(f"fake mongo: update operator {op}")


class _Result:
    def __init__(self, inserted_id=None, deleted_count=0, modified_count=0):
        self.inserted_id = inserted_id
        self.deleted_count = deleted_count
        self.modified_count = modified_count


class _Cursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, field, direction=1):
        self._docs.sort(key=lambda d: d.get(field), reverse=direction < 0)
        return self

    def limit(self, count):
        self._docs = self._docs[:count]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    """One lock per collection, held across each operation.

    That is the property being modelled: MongoDB evaluates the filter of an
    update and applies the modification as one indivisible step. A test that
    used a lock-free dict would pass even if db.py did an unsafe read-then-write.
    """

    def __init__(self):
        self._docs: dict = {}
        self._unique: list = []
        self._lock = threading.RLock()

    def create_index(self, keys, unique=False, **_kwargs):
        if unique:
            self._unique.extend(field for field, _direction in keys)
        return "_".join(f"{f}_{d}" for f, d in keys)

    def insert_one(self, doc):
        with self._lock:
            doc = copy.deepcopy(doc)
            doc.setdefault("_id", ObjectId())
            if doc["_id"] in self._docs:
                raise DuplicateKeyError(f"duplicate _id {doc['_id']}")
            for field in self._unique:
                if any(existing.get(field) == doc.get(field)
                       for existing in self._docs.values()):
                    raise DuplicateKeyError(f"duplicate {field}")
            self._docs[doc["_id"]] = doc
            return _Result(inserted_id=doc["_id"])

    def find_one(self, query, projection=None):
        with self._lock:
            for doc in self._docs.values():
                if _matches(doc, query):
                    return copy.deepcopy(doc)
            return None

    def find(self, query):
        with self._lock:
            return _Cursor([copy.deepcopy(d) for d in self._docs.values()
                            if _matches(d, query)])

    def find_one_and_update(self, query, update, return_document=True):
        with self._lock:
            for doc in self._docs.values():
                if _matches(doc, query):
                    before = copy.deepcopy(doc)
                    _apply(doc, update)
                    return copy.deepcopy(doc) if return_document else before
            return None

    def update_one(self, query, update):
        with self._lock:
            for doc in self._docs.values():
                if _matches(doc, query):
                    _apply(doc, update)
                    return _Result(modified_count=1)
            return _Result()

    def delete_one(self, query):
        with self._lock:
            for key, doc in list(self._docs.items()):
                if _matches(doc, query):
                    del self._docs[key]
                    return _Result(deleted_count=1)
            return _Result()

    def count(self):
        with self._lock:
            return len(self._docs)


class FakeDatabase:
    def __init__(self):
        self._collections: dict = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._collections.setdefault(name, FakeCollection())

    __getitem__ = __getattr__


class FakeClient:
    def __init__(self, *_args, **_kwargs):
        self._databases: dict = {}

    @property
    def admin(self):
        return types.SimpleNamespace(command=lambda *_a, **_k: {"ok": 1})

    def __getitem__(self, name):
        return self._databases.setdefault(name, FakeDatabase())


class ReturnDocument:
    BEFORE = False
    AFTER = True


def install_fakes() -> None:
    """Put the stand-ins in sys.modules so the real modules import cleanly."""
    pymongo = types.ModuleType("pymongo")
    pymongo.ASCENDING = 1
    pymongo.DESCENDING = -1
    pymongo.MongoClient = FakeClient
    pymongo.ReturnDocument = ReturnDocument
    pymongo.version = "fake"
    errors = types.ModuleType("pymongo.errors")
    errors.DuplicateKeyError = DuplicateKeyError
    errors.PyMongoError = Exception
    pymongo.errors = errors

    bson = types.ModuleType("bson")
    bson.ObjectId = ObjectId
    bson_errors = types.ModuleType("bson.errors")
    bson_errors.InvalidId = InvalidId
    bson.errors = bson_errors

    for name, module in (("pymongo", pymongo), ("pymongo.errors", errors),
                         ("bson", bson), ("bson.errors", bson_errors)):
        sys.modules.setdefault(name, module)

    # fastapi and pydantic are only needed for backend/auth.py to import; the
    # request handling under test is plain Python.
    if "fastapi" not in sys.modules:
        fastapi = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str = ""):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        def _decorator(*_a, **_k):
            return lambda fn: fn

        class _Router:
            def __init__(self, *_a, **_k):
                pass

            post = get = delete = put = staticmethod(_decorator)

        fastapi.APIRouter = _Router
        fastapi.HTTPException = HTTPException
        fastapi.Cookie = lambda default=None, **_k: default
        fastapi.Depends = lambda dependency=None: dependency
        fastapi.Request = object
        fastapi.Response = object
        fastapi.Form = lambda default=None, **_k: default
        fastapi.UploadFile = object
        fastapi.File = lambda default=None, **_k: default
        sys.modules["fastapi"] = fastapi

    if "pydantic" not in sys.modules:
        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                for key, value in type(self).__dict__.items():
                    if not key.startswith("_") and not hasattr(self, key):
                        setattr(self, key, value)

        pydantic.BaseModel = BaseModel
        sys.modules["pydantic"] = pydantic


install_fakes()

from backend import auth                                              # noqa: E402
from backend import db                                                # noqa: E402


# --------------------------------------------------------------------------- #
# Test-side helpers
# --------------------------------------------------------------------------- #

class FakeRequest:
    """Enough of starlette's Request for auth.current_user and _set_cookie."""

    def __init__(self, headers=None, scheme="http"):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.url = types.SimpleNamespace(scheme=scheme)


class FakeResponse:
    def __init__(self):
        self.cookies: dict = {}
        self.deleted: list = []

    def set_cookie(self, name, value, **kwargs):
        self.cookies[name] = {"value": value, **kwargs}

    def delete_cookie(self, name, **_kwargs):
        self.deleted.append(name)
        self.cookies.pop(name, None)


def fresh_database():
    """Reset db.py onto an empty fake and re-run its real connect()."""
    db._client = None
    db._db = None
    db._init_error = None
    db.connect()
    return db._db


def make_user(email="a@b.com", name="Test User", password="password123"):
    return db.create_user(name, email, auth.hash_password(password))


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        fresh_database()
        self.addCleanup(setattr, db, "_today", db._today)


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #

class TestPasswords(unittest.TestCase):

    def test_hash_is_not_the_password(self):
        stored = auth.hash_password("correct horse battery")
        self.assertNotIn("correct horse battery", stored)
        self.assertTrue(stored.startswith("scrypt$"))

    def test_same_password_hashes_differently_each_time(self):
        # A per-password salt: two users with the same password must not share a
        # hash, or one cracked hash would unlock both accounts.
        first = auth.hash_password("password123")
        second = auth.hash_password("password123")
        self.assertNotEqual(first, second)
        self.assertTrue(auth.verify_password("password123", first))
        self.assertTrue(auth.verify_password("password123", second))

    def test_wrong_password_is_rejected(self):
        stored = auth.hash_password("password123")
        for attempt in ("password124", "Password123", "", "password1234",
                        "password12"):
            self.assertFalse(auth.verify_password(attempt, stored), attempt)

    def test_malformed_stored_value_is_a_failed_login_not_a_crash(self):
        for stored in ("", "not-a-hash", "scrypt$only$three$parts",
                       "bcrypt$1$2$3$4$5", "scrypt$x$y$z$zz$zz", None):
            self.assertFalse(auth.verify_password("password123", stored),
                             repr(stored))


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

class TestTokens(unittest.TestCase):

    def test_round_trip(self):
        token = auth.make_token("64b7f0c2e1a2b3c4d5e6f708")
        self.assertEqual(auth.read_token(token), "64b7f0c2e1a2b3c4d5e6f708")

    def test_token_is_a_string(self):
        # PyJWT 1.x returned bytes, which set_cookie rejects far from here.
        self.assertIsInstance(auth.make_token("x"), str)

    def test_tampered_token_is_refused(self):
        token = auth.make_token("64b7f0c2e1a2b3c4d5e6f708")
        head, body, sig = token.split(".")
        forged = auth.make_token("000000000000000000000001").split(".")[1]
        self.assertIsNone(auth.read_token(f"{head}.{forged}.{sig}"))
        self.assertIsNone(auth.read_token(f"{head}.{body}.{sig[:-2]}xx"))
        self.assertIsNone(auth.read_token(""))
        self.assertIsNone(auth.read_token("garbage"))

    def test_token_signed_with_another_secret_is_refused(self):
        original = auth.SECRET
        try:
            auth.SECRET = "a-different-secret"
            other = auth.make_token("64b7f0c2e1a2b3c4d5e6f708")
        finally:
            auth.SECRET = original
        self.assertIsNone(auth.read_token(other))

    def test_expired_token_is_refused(self):
        from datetime import timedelta
        original = auth.TOKEN_TTL
        try:
            auth.TOKEN_TTL = timedelta(seconds=-10)
            stale = auth.make_token("64b7f0c2e1a2b3c4d5e6f708")
        finally:
            auth.TOKEN_TTL = original
        self.assertIsNone(auth.read_token(stale))


# --------------------------------------------------------------------------- #
# Registration and login
# --------------------------------------------------------------------------- #

class TestRegistration(DatabaseTestCase):

    def register(self, **kwargs):
        payload = auth.RegisterIn(**kwargs)
        return auth.register(payload, FakeRequest(), FakeResponse())

    def test_new_account_starts_with_a_full_day_of_credits(self):
        body = self.register(name="Sam", email="sam@example.com",
                             password="password123",
                             confirm_password="password123")
        self.assertEqual(body["credits_remaining"], 15)
        self.assertEqual(body["credits_total"], 15)
        self.assertEqual(body["videos_per_day"], 3)
        self.assertTrue(body["can_process"])

    def test_response_never_contains_the_password_or_its_hash(self):
        body = self.register(name="Sam", email="sam@example.com",
                             password="password123",
                             confirm_password="password123")
        flat = repr(body)
        self.assertNotIn("password123", flat)
        self.assertNotIn("scrypt$", flat)
        self.assertNotIn("password_hash", body["user"])

    def test_stored_record_holds_a_hash_and_not_the_password(self):
        self.register(name="Sam", email="sam@example.com",
                      password="password123", confirm_password="password123")
        stored = db.find_by_email("sam@example.com")
        self.assertNotIn("password", stored)
        self.assertTrue(stored["password_hash"].startswith("scrypt$"))
        self.assertNotIn("password123", stored["password_hash"])

    def test_email_is_stored_lowercase_so_case_cannot_duplicate_an_account(self):
        self.register(name="Sam", email="  SAM@Example.COM ",
                      password="password123", confirm_password="password123")
        self.assertIsNotNone(db.find_by_email("sam@example.com"))
        with self.assertRaises(auth.HTTPException) as caught:
            self.register(name="Imposter", email="sam@example.com",
                          password="password456", confirm_password="password456")
        self.assertEqual(caught.exception.status_code, 409)

    def test_invalid_registrations_are_refused_with_400(self):
        cases = [
            dict(name="", email="a@b.com", password="password123",
                 confirm_password="password123"),
            dict(name="Sam", email="", password="password123",
                 confirm_password="password123"),
            dict(name="Sam", email="not-an-email", password="password123",
                 confirm_password="password123"),
            dict(name="Sam", email="a@b", password="password123",
                 confirm_password="password123"),
            dict(name="Sam", email="a@b.com", password="short",
                 confirm_password="short"),
            dict(name="Sam", email="a@b.com", password="password123",
                 confirm_password="password124"),
        ]
        for case in cases:
            with self.subTest(**case):
                with self.assertRaises(auth.HTTPException) as caught:
                    self.register(**case)
                self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(db.get_db().users.count(), 0)


class TestLogin(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        auth.register(auth.RegisterIn(name="Sam", email="sam@example.com",
                                      password="password123",
                                      confirm_password="password123"),
                      FakeRequest(), FakeResponse())

    def login(self, email, password, request=None):
        response = FakeResponse()
        body = auth.login(auth.LoginIn(email=email, password=password),
                          request or FakeRequest(), response)
        return body, response

    def test_correct_credentials_return_the_user_and_credits(self):
        body, response = self.login("sam@example.com", "password123")
        self.assertEqual(body["user"]["email"], "sam@example.com")
        self.assertEqual(body["credits_remaining"], 15)
        self.assertIn(auth.COOKIE_NAME, response.cookies)

    def test_session_cookie_is_httponly_and_not_readable_by_javascript(self):
        _, response = self.login("sam@example.com", "password123")
        cookie = response.cookies[auth.COOKIE_NAME]
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "lax")
        self.assertEqual(cookie["path"], "/")

    def test_secure_flag_follows_the_scheme(self):
        # Hardcoding secure=True would make the cookie silently vanish over
        # plain http on a LAN, so login would appear to work and never stick.
        _, plain = self.login("sam@example.com", "password123")
        self.assertFalse(plain.cookies[auth.COOKIE_NAME]["secure"])
        _, tls = self.login("sam@example.com", "password123",
                            FakeRequest(scheme="https"))
        self.assertTrue(tls.cookies[auth.COOKIE_NAME]["secure"])
        _, proxied = self.login("sam@example.com", "password123",
                                FakeRequest({"x-forwarded-proto": "https"}))
        self.assertTrue(proxied.cookies[auth.COOKIE_NAME]["secure"])

    def test_wrong_password_and_unknown_email_are_indistinguishable(self):
        with self.assertRaises(auth.HTTPException) as wrong:
            self.login("sam@example.com", "password124")
        with self.assertRaises(auth.HTTPException) as missing:
            self.login("nobody@example.com", "password123")
        self.assertEqual(wrong.exception.status_code, 401)
        self.assertEqual(missing.exception.status_code, 401)
        # Identical text, so the endpoint cannot be used to discover which
        # addresses have accounts.
        self.assertEqual(wrong.exception.detail, missing.exception.detail)

    def test_logout_clears_the_cookie(self):
        response = FakeResponse()
        auth.logout(response)
        self.assertIn(auth.COOKIE_NAME, response.deleted)


# --------------------------------------------------------------------------- #
# The session dependency
# --------------------------------------------------------------------------- #

class TestCurrentUser(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.user = make_user()
        self.token = auth.make_token(self.user["_id"])

    def test_cookie_is_accepted(self):
        found = auth.current_user(FakeRequest(), self.token)
        self.assertEqual(found["email"], "a@b.com")

    def test_bearer_header_is_accepted(self):
        request = FakeRequest({"Authorization": f"Bearer {self.token}"})
        self.assertEqual(auth.current_user(request, None)["email"], "a@b.com")

    def test_no_credentials_is_401(self):
        with self.assertRaises(auth.HTTPException) as caught:
            auth.current_user(FakeRequest(), None)
        self.assertEqual(caught.exception.status_code, 401)

    def test_junk_and_wrong_scheme_are_401(self):
        for headers, cookie in (({}, "not-a-token"),
                                ({"Authorization": "Basic abc"}, None),
                                ({"Authorization": "Bearer "}, None)):
            with self.subTest(headers=headers, cookie=cookie):
                with self.assertRaises(auth.HTTPException) as caught:
                    auth.current_user(FakeRequest(headers), cookie)
                self.assertEqual(caught.exception.status_code, 401)

    def test_token_for_a_deleted_account_is_401_not_a_crash(self):
        db.get_db().users.delete_one({"_id": self.user["_id"]})
        with self.assertRaises(auth.HTTPException) as caught:
            auth.current_user(FakeRequest(), self.token)
        self.assertEqual(caught.exception.status_code, 401)

    def test_401_is_used_throughout_never_403(self):
        # The frontend treats 401 as "session gone, show the login page". A 403
        # anywhere in this path would leave it stuck instead.
        source = (ROOT / "backend" / "auth.py").read_text(encoding="utf-8")
        self.assertNotIn("status_code=403", source)


# --------------------------------------------------------------------------- #
# Credits
# --------------------------------------------------------------------------- #

class TestCredits(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.user = make_user()
        self.uid = self.user["_id"]

    def test_three_videos_spend_the_day_and_the_fourth_is_refused(self):
        for expected in (10, 5, 0):
            outcome = db.reserve_credits(self.uid, f"req-{expected}")
            self.assertTrue(outcome["ok"])
            self.assertEqual(outcome["remaining"], expected)

        fourth = db.reserve_credits(self.uid, "req-fourth")
        self.assertFalse(fourth["ok"])
        self.assertEqual(fourth["reason"], "insufficient")
        self.assertEqual(fourth["remaining"], 0)
        # The balance must not have gone negative, and the refused request must
        # not have left a charge behind.
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 0)
        self.assertEqual(db.get_db().credit_charges.count(), 3)

    def test_credit_state_tracks_videos_used(self):
        self.assertEqual(db.credit_state(self.uid)["videos_today"], 0)
        db.reserve_credits(self.uid, "r1")
        state = db.credit_state(self.uid)
        self.assertEqual(state["videos_today"], 1)
        self.assertEqual(state["credits_remaining"], 10)
        self.assertTrue(state["can_process"])
        db.reserve_credits(self.uid, "r2")
        db.reserve_credits(self.uid, "r3")
        state = db.credit_state(self.uid)
        self.assertEqual(state["videos_today"], 3)
        self.assertFalse(state["can_process"])

    def test_a_repeated_request_id_is_not_charged_twice(self):
        first = db.reserve_credits(self.uid, "same-id")
        second = db.reserve_credits(self.uid, "same-id")
        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["reason"], "duplicate")
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 10)

    def test_two_users_are_not_confused_by_the_same_request_id(self):
        # app.py namespaces the charge id with the user id for exactly this
        # reason: otherwise one account could burn another account's ids.
        other = make_user(email="other@b.com")
        first = db.reserve_credits(self.uid, f"{self.uid}:shared")
        second = db.reserve_credits(other["_id"], f"{other['_id']}:shared")
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 10)
        self.assertEqual(db.credit_state(other["_id"])["credits_remaining"], 10)

    def test_concurrent_uploads_cannot_overspend(self):
        """Six simultaneous requests against a 15-credit balance: exactly three
        may be charged. This is the case a read-then-write check would fail."""
        outcomes = []
        lock = threading.Lock()
        start = threading.Event()

        def attempt(index):
            start.wait(timeout=5)
            result = db.reserve_credits(self.uid, f"burst-{index}")
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(len(outcomes), 6)
        self.assertEqual(sum(1 for o in outcomes if o["ok"]), 3)
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 0)
        self.assertTrue(all(o["remaining"] >= 0 for o in outcomes))

    def test_a_failed_job_is_refunded_once(self):
        db.reserve_credits(self.uid, "will-fail")
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 10)
        self.assertEqual(db.refund_credits(self.uid, "will-fail"), 15)
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 15)
        # Idempotent: a second refund of the same charge returns nothing and
        # cannot inflate the balance.
        self.assertIsNone(db.refund_credits(self.uid, "will-fail"))
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 15)

    def test_a_refund_cannot_push_the_balance_above_the_daily_total(self):
        db.reserve_credits(self.uid, "one")
        db.get_db().users.update_one({"_id": self.uid},
                                     {"$set": {"daily_credits": 15}})
        self.assertEqual(db.refund_credits(self.uid, "one"), 15)

    def test_refunding_an_unknown_charge_does_nothing(self):
        self.assertIsNone(db.refund_credits(self.uid, "never-charged"))
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 15)


class TestDailyReset(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.user = make_user()
        self.uid = self.user["_id"]

    def test_credits_come_back_on_the_next_day(self):
        for i in range(3):
            db.reserve_credits(self.uid, f"day1-{i}")
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 0)

        # No scheduler: the reset happens on the first request of the new day,
        # so moving the clock forward is all it takes.
        db._today = lambda: "2999-01-01"
        state = db.credit_state(self.uid)
        self.assertEqual(state["credits_remaining"], 15)
        self.assertEqual(state["videos_today"], 0)
        self.assertEqual(
            db.get_db().users.find_one({"_id": self.uid})["last_credit_reset"],
            "2999-01-01")

    def test_the_reset_happens_only_once_a_day(self):
        db.reserve_credits(self.uid, "a")
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 10)
        # Repeated reads on the same day must not keep topping the balance up.
        for _ in range(5):
            db.credit_state(self.uid)
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 10)

    def test_concurrent_first_requests_of_the_day_reset_only_once(self):
        db.reserve_credits(self.uid, "yesterday")
        db.get_db().users.update_one(
            {"_id": self.uid}, {"$set": {"last_credit_reset": "2000-01-01"}})

        start = threading.Event()

        def attempt():
            start.wait(timeout=5)
            db.apply_daily_reset(self.uid)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=10)

        # 15, not 15 + 15: the condition lives in the update filter.
        self.assertEqual(db.credit_state(self.uid)["credits_remaining"], 15)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

class TestHistory(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.user = make_user()
        self.uid = self.user["_id"]
        self.other = make_user(email="other@b.com")

    def record(self, job_id, name):
        db.record_job(job_id, self.uid, f"{self.uid}:{job_id}", name,
                      f"job_{job_id}_annotated.mp4", {"accidents": True})

    def test_a_row_is_written_per_job_and_holds_no_video_bytes(self):
        self.record("j1", "clip.mp4")
        row = db.get_db().video_history.find_one({"_id": "j1"})
        self.assertEqual(row["status"], "processing")
        self.assertEqual(row["credits_used"], 5)
        self.assertEqual(row["original_filename"], "clip.mp4")
        # Metadata only: the row points at a filename, it does not contain one.
        for key, value in row.items():
            self.assertNotIsInstance(value, (bytes, bytearray), key)

    def test_finishing_a_job_records_the_outcome(self):
        self.record("j1", "clip.mp4")
        db.finish_job("j1", "completed", summary={"vehicles": 12})
        row = db.get_db().video_history.find_one({"_id": "j1"})
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["summary"], {"vehicles": 12})
        self.assertIsNotNone(row["finished_at"])

    def test_a_failed_job_is_recorded_as_costing_nothing(self):
        self.record("j1", "clip.mp4")
        db.finish_job("j1", "failed", error="decode error", credits_used=0)
        row = db.get_db().video_history.find_one({"_id": "j1"})
        self.assertEqual(row["credits_used"], 0)
        self.assertEqual(row["error"], "decode error")

    def test_history_is_newest_first_and_only_the_owner_sees_it(self):
        for i in range(3):
            self.record(f"j{i}", f"clip{i}.mp4")
        db.record_job("other-job", self.other["_id"], "x", "theirs.mp4",
                      "out.mp4", {})

        rows = db.history_for(self.uid)
        self.assertEqual(len(rows), 3)
        self.assertNotIn("theirs.mp4", [r["original_filename"] for r in rows])
        times = [r["created_at"] for r in rows]
        self.assertEqual(times, sorted(times, reverse=True))

    def test_job_owner_identifies_the_uploader(self):
        self.record("j1", "clip.mp4")
        self.assertEqual(db.job_owner("j1"), str(self.uid))
        self.assertNotEqual(db.job_owner("j1"), str(self.other["_id"]))
        self.assertIsNone(db.job_owner("no-such-job"))


# --------------------------------------------------------------------------- #
# Every API route that touches user data requires a session
# --------------------------------------------------------------------------- #

class TestEndpointsAreProtected(unittest.TestCase):
    """Read backend/app.py as source, so a new unprotected route fails here.

    app.py cannot be imported without fastapi and torch, and this does not need
    to run it: whether a route depends on auth.current_user is visible in the
    signature.
    """

    # Two deliberately open routes, neither of which touches an account:
    # /api/health reports whether the server and database are up plus the credit
    # limits the landing page quotes, and "/" is the frontend - the login page
    # has to be reachable by someone who is not logged in.
    PUBLIC = {"/api/health", "/"}

    @staticmethod
    def routes():
        tree = ast.parse((ROOT / "backend" / "app.py").read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                target = decorator.func
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "app"
                        and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)):
                    found.append((decorator.args[0].value, node))
        return found

    def test_at_least_the_known_routes_were_found(self):
        paths = {path for path, _ in self.routes()}
        for expected in ("/api/process", "/api/status/{job_id}",
                         "/api/result/{job_id}", "/api/result/{job_id}/video",
                         "/api/result/{job_id}/download", "/api/history",
                         "/api/jobs", "/api/cancel/{job_id}"):
            self.assertIn(expected, paths)

    def test_every_non_public_route_depends_on_current_user(self):
        for path, node in self.routes():
            if path in self.PUBLIC:
                continue
            with self.subTest(path=path):
                sources = [ast.unparse(default)
                           for default in node.args.defaults if default]
                self.assertTrue(
                    any("current_user" in text for text in sources),
                    f"{path} does not require a signed-in user")


if __name__ == "__main__":
    unittest.main(verbosity=2)
