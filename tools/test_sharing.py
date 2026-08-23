"""Unit tests for the access gate and disk-retention logic.

Runs without torch, ultralytics, fastapi or a GPU:

    python tools/test_sharing.py

These cover the properties that matter once the app is reachable by other people:
that an unauthenticated public bind is refused outright, that token comparison
cannot be brute-forced or bypassed, and that retained job files stay bounded so a
shared URL cannot quietly fill the host's disk.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.jobs import JobManager                                   # noqa: E402


def load_security_helpers():
    """Import the pure helpers from backend/security.py without fastapi.

    security.py imports fastapi for its routes, but the policy functions tested
    here are dependency-free. Stripping the fastapi imports keeps this suite
    runnable in a bare checkout, which is the whole point of the tools/ tests.
    """
    src = (ROOT / "backend" / "security.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    kept = [n for n in tree.body
            if not (isinstance(n, (ast.Import, ast.ImportFrom))
                    and "fastapi" in ast.unparse(n))]
    module = types.ModuleType("security_pure")
    exec(compile(ast.Module(body=kept, type_ignores=[]), "security", "exec"),
         module.__dict__)
    return module


sec = load_security_helpers()


class TestBindPolicy(unittest.TestCase):
    """An unauthenticated server must never come up on a public interface."""

    def test_loopback_addresses_are_recognised(self):
        for host in ("127.0.0.1", "localhost", "::1", "127.5.5.5"):
            self.assertTrue(sec.is_loopback(host), host)

    def test_public_and_unparseable_hosts_are_not_loopback(self):
        # An empty host means "all interfaces" to uvicorn, and an unresolvable
        # name cannot be proven local, so both must be treated as public. Getting
        # this backwards would be a silent hole rather than a visible error.
        for host in ("0.0.0.0", "", "::", "192.168.1.50", "10.0.0.7", "example.com"):
            self.assertFalse(sec.is_loopback(host), host)

    def test_public_bind_without_token_is_refused(self):
        for host in ("0.0.0.0", "192.168.1.50", ""):
            with self.assertRaises(SystemExit, msg=host):
                sec.enforce_bind_policy(host, None)

    def test_public_bind_with_token_is_allowed(self):
        sec.enforce_bind_policy("0.0.0.0", "a-token")      # must not raise

    def test_loopback_without_token_is_allowed(self):
        """Local development must keep working with no configuration."""
        sec.enforce_bind_policy("127.0.0.1", None)         # must not raise

    def test_refusal_message_names_the_env_var(self):
        """The error has to be actionable, not just a refusal."""
        with self.assertRaises(SystemExit) as ctx:
            sec.enforce_bind_policy("0.0.0.0", None)
        self.assertIn(sec.ENV_VAR, str(ctx.exception))


class TestTokenComparison(unittest.TestCase):
    def test_exact_token_accepted(self):
        self.assertTrue(sec.token_is_valid("s3cret", "s3cret"))

    def test_wrong_tokens_rejected(self):
        for supplied in ("S3cret", "s3cre", "s3cret ", " s3cret", "", None,
                         "s3cretx"):
            self.assertFalse(sec.token_is_valid(supplied, "s3cret"), repr(supplied))

    def test_empty_supplied_never_matches_even_empty_token(self):
        """Guards against an unset token degrading into "any request passes"."""
        self.assertFalse(sec.token_is_valid(None, ""))
        self.assertFalse(sec.token_is_valid("", ""))

    def test_login_is_not_gated_by_the_gate(self):
        """/login must stay reachable or nobody could ever authenticate."""
        self.assertIn("/login", sec.PUBLIC_PATHS)

    def test_health_is_not_public(self):
        """/api/health reports device and model paths, so it needs the token.
        /api/ping exists for liveness checks instead."""
        self.assertNotIn("/api/health", sec.PUBLIC_PATHS)
        self.assertIn("/api/ping", sec.PUBLIC_PATHS)


class TestLoginThrottle(unittest.TestCase):
    def setUp(self):
        sec._failures.clear()

    def test_lockout_engages_after_repeated_failures(self):
        key = "203.0.113.9"
        for _ in range(sec.MAX_FAILURES):
            self.assertEqual(sec._locked_out(key), 0.0)
            sec._record_failure(key)
        self.assertGreater(sec._locked_out(key), 0.0)

    def test_success_clears_the_failure_record(self):
        key = "203.0.113.10"
        for _ in range(sec.MAX_FAILURES):
            sec._record_failure(key)
        sec._clear_failures(key)
        self.assertEqual(sec._locked_out(key), 0.0)

    def test_lockout_is_per_client(self):
        for _ in range(sec.MAX_FAILURES):
            sec._record_failure("198.51.100.1")
        self.assertEqual(sec._locked_out("198.51.100.2"), 0.0)


class TestRetention(unittest.TestCase):
    """A shared URL must not be able to fill the host's disk."""

    def _run(self, max_history, max_bytes, mb_each, count):
        tmp = Path(tempfile.mkdtemp())
        mgr = JobManager(runner=lambda job: {}, max_history=max_history,
                         max_bytes=max_bytes)
        for i in range(count):
            inp = tmp / f"j{i}_in.mp4"
            inp.write_bytes(b"\0" * (mb_each * 1024 * 1024))
            out = tmp / f"j{i}_out.mp4"
            out.write_bytes(b"\0" * 1024)
            mgr.submit(input_path=inp, output_path=out,
                       original_filename=f"c{i}.mp4", options={}, job_id=f"j{i}")
            time.sleep(0.25)
        time.sleep(0.4)
        on_disk = sum(p.stat().st_size for p in tmp.glob("*") if p.is_file())
        return mgr, on_disk

    def test_evicted_jobs_have_their_files_deleted(self):
        mgr, on_disk = self._run(max_history=2, max_bytes=8 * 1024 ** 3,
                                 mb_each=1, count=5)
        self.assertEqual(len(mgr.list()), 2)
        # 5 MB written, at most the 2 retained jobs remain.
        self.assertLess(on_disk, 4 * 1024 * 1024)

    def test_byte_budget_evicts_even_when_count_is_fine(self):
        """max_history alone is not a disk bound: 50 retained jobs at the 2 GB
        upload limit is 100 GB, so bytes must be capped independently."""
        budget = 50 * 1024 * 1024
        mgr, on_disk = self._run(max_history=50, max_bytes=budget,
                                 mb_each=20, count=5)
        self.assertLessEqual(on_disk, budget * 1.25)
        self.assertLess(len(mgr.list()), 5)

    def test_live_jobs_are_never_evicted(self):
        """Deleting a queued job's upload would break a run that has not started."""
        import threading
        release = threading.Event()
        tmp = Path(tempfile.mkdtemp())
        mgr = JobManager(runner=lambda job: release.wait(timeout=10),
                         max_history=1, max_bytes=1024)
        paths = []
        for i in range(4):
            inp = tmp / f"j{i}_in.mp4"
            inp.write_bytes(b"\0" * 4096)
            out = tmp / f"j{i}_out.mp4"
            out.write_bytes(b"\0" * 4096)
            paths += [inp, out]
            mgr.submit(input_path=inp, output_path=out,
                       original_filename=f"c{i}.mp4", options={}, job_id=f"j{i}")
        time.sleep(0.4)
        try:
            # Far over both ceilings, but nothing has finished, so nothing may go.
            self.assertTrue(all(p.exists() for p in paths))
            self.assertEqual(len(mgr.list()), 4)
        finally:
            release.set()

    def test_pending_counts_only_queued_work(self):
        import threading
        release = threading.Event()
        tmp = Path(tempfile.mkdtemp())
        mgr = JobManager(runner=lambda job: release.wait(timeout=10))
        for i in range(3):
            inp = tmp / f"p{i}_in.mp4"
            inp.write_bytes(b"x")
            out = tmp / f"p{i}_out.mp4"
            out.write_bytes(b"x")
            mgr.submit(input_path=inp, output_path=out,
                       original_filename=f"c{i}.mp4", options={}, job_id=f"p{i}")
        time.sleep(0.4)
        try:
            # One is processing, so two are pending - the API caps on this number.
            self.assertEqual(mgr.pending(), 2)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main(verbosity=2)
