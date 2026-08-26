#!/usr/bin/env python3
"""tests/test_downloader.py — Python driver for the C++ downloader test (TOOLS-5).

Spins up the local registry server (`tools/serve_registry.py`), then
runs the C++ `test_downloader` binary against a controlled
{model_dir, cache_dir, registry} fixture, and validates the stdout /
exit code. The scenarios covered:

  - registry         : parse + entry lookup
  - noop             : empty name
  - missing          : missing registry entry
  - local_ok         : file present and matching
  - local_corrupt_offline : offline + corrupt -> ERR_MODEL
  - local_corrupt_download : corrupt -> re-download -> OK
  - offline_missing  : offline + file missing -> ERR_MODEL
  - download_no_curl : only runs when curl is missing from the build

The test also runs the C++ test with `--offline 0 --download 1` and
checks the actual bytes on disk in model_dir/cache_dir after the run.

This test is self-contained: it builds a 10KB random-byte "model"
and a matching registry.json, starts the server, and tears it down
on exit. No fixtures outside `/tmp` are touched.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The worktree's build dir; defaults to the env var set by CI / developer.
BUILD_DIR = Path(os.environ.get("PPOCR_BUILD_DIR",
                                 REPO_ROOT / "build-tools"))
# pp-ocr-mnn source tree. The worktree lives at <root>/<worktree>;
# pp-ocr-mnn is a sibling of the worktree, i.e. at <root>/pp-ocr-mnn.
# Allow override via env var for unusual layouts.
PPOCR_MNN_ROOT = Path(os.environ.get("PPOCR_MNN_ROOT",
                                      REPO_ROOT.parent.parent / "pp-ocr-mnn"))
TEST_BIN = BUILD_DIR / "test_downloader"
SERVE = REPO_ROOT / "tools" / "serve_registry.py"


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _allocate_port() -> int:
    for p in range(9900, 9999):
        if _port_free(p):
            return p
    raise RuntimeError("no free port in 9900..9999")


@contextlib.contextmanager
def _serve_registry(mirror: Path, port: int):
    """Start serve_registry.py as a subprocess and yield its process."""
    log_path = mirror / "serve.log"
    f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(SERVE), "--root", str(mirror),
         "--port", str(port), "--bind", "127.0.0.1"],
        stdout=f, stderr=subprocess.STDOUT,
        # New process group so we can SIGTERM the children.
        start_new_session=True,
    )
    # Wait for the port to accept connections (up to 5s)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if _port_free(port):
            time.sleep(0.1)
            continue
        # Port is in use -> the server is listening.
        break
    else:
        proc.terminate()
        raise RuntimeError(f"serve_registry did not start on port {port} "
                           f"in 5s; see {log_path}")
    # Hit /healthz to confirm it's actually serving (not just bound)
    for _ in range(20):
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                        timeout=1) as r:
                if r.status == 200 and r.read().decode() == "ok":
                    break
        except Exception:
            time.sleep(0.1)
    else:
        proc.terminate()
        raise RuntimeError("serve_registry /healthz did not respond")
    try:
        yield proc
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        f.close()


def _make_fixture(root: Path) -> dict:
    """Create a small registry + one .mnn under `root / mirror`.

    Returns a dict with paths so the test can refer to them."""
    mirror = root / "mirror"
    model_dir = root / "model_dir"
    cache_dir = root / "cache_dir"
    mirror.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 10KB of deterministic random-ish bytes (same algo as C++ test)
    random_data = bytes((i * 17 + 31) & 0xff for i in range(10_000))
    (mirror / "my_test_model.mnn").write_bytes(random_data)
    sha = hashlib.sha256(random_data).hexdigest()
    reg = {
        "my_test_model": {
            "name": "my_test_model",
            "type": "det",
            "file": "my_test_model.mnn",
            "sha256": sha,
            "bytes": len(random_data),
            "url": "my_test_model.mnn",
        },
    }
    (mirror / "registry.json").write_text(json.dumps(reg, indent=2))
    return {
        "mirror": mirror,
        "model_dir": model_dir,
        "cache_dir": cache_dir,
        "registry": mirror / "registry.json",
        "sha": sha,
        "size": len(random_data),
    }


class TestServeRegistry(unittest.TestCase):
    """Smoke test for the Python server: endpoints + sha helper."""

    def test_serve_endpoints(self):
        if not SERVE.exists():
            self.skipTest(f"serve_registry.py missing at {SERVE}")
        port = _allocate_port()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x.mnn").write_bytes(b"hello")
            with _serve_registry(root, port):
                import urllib.request
                # /healthz
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/healthz") as r:
                    self.assertEqual(r.read().decode(), "ok")
                # /x.mnn
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/x.mnn") as r:
                    self.assertEqual(r.read(), b"hello")
                # /sha256/x.mnn
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/sha256/x.mnn") as r:
                    self.assertEqual(r.read().decode(),
                                     hashlib.sha256(b"hello").hexdigest())
                # / (index)
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/") as r:
                    self.assertIn(b"x.mnn", r.read())
                # 404 for missing
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/missing.mnn") as r:
                        self.assertEqual(r.status, 404)
                except urllib.error.HTTPError as e:
                    self.assertEqual(e.code, 404)


class TestDownloader(unittest.TestCase):
    """End-to-end downloader test (TOOLS-5 / M3)."""

    @classmethod
    def setUpClass(cls):
        if not TEST_BIN.exists():
            raise unittest.SkipTest(
                f"test_downloader not built at {TEST_BIN} (set "
                f"PPOCR_BUILD_DIR or run cmake --build first)")

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="dltest_"))
        self.fx = _make_fixture(self.workdir)
        # Place a copy of the model in model_dir so the "local_ok"
        # scenario has something to verify.
        shutil.copy(self.fx["mirror"] / "my_test_model.mnn",
                    self.fx["model_dir"])
        # Allocate a free port for serve_registry
        self.port = _allocate_port()

    def tearDown(self):
        # Best-effort cleanup; on Windows the lock can linger but we're
        # on Linux/macOS where rm -rf is reliable.
        try:
            shutil.rmtree(self.workdir)
        except OSError:
            pass

    def _run_cpp(self, *args) -> subprocess.CompletedProcess:
        cmd = [str(TEST_BIN),
               "--registry", str(self.fx["registry"]),
               "--model-dir", str(self.fx["model_dir"]),
               "--cache-dir", str(self.fx["cache_dir"]),
               "--mirror", f"http://127.0.0.1:{self.port}",
               *args]
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30, env={**os.environ,
                                              "LD_LIBRARY_PATH":
                                              "/usr/lib/x86_64-linux-gnu"})

    def test_serve_registry_present(self):
        self.assertTrue(SERVE.exists(),
                        f"serve_registry.py missing at {SERVE}")

    def test_serve_starts_and_serves_registry(self):
        with _serve_registry(self.fx["mirror"], self.port):
            import urllib.request
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/registry.json") as r:
                body = json.loads(r.read())
            self.assertIn("my_test_model", body)
            self.assertEqual(body["my_test_model"]["sha256"], self.fx["sha"])
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/my_test_model.mnn") as r:
                self.assertEqual(len(r.read()), self.fx["size"])

    def test_cpp_test_all_scenarios(self):
        """The C++ test driver must pass all scenarios against the
        real serve_registry.py server."""
        with _serve_registry(self.fx["mirror"], self.port):
            # Run the C++ binary in offline mode (should pass)
            p = self._run_cpp("--offline", "1", "--download", "0",
                              "--scenario", "all")
            self.assertEqual(p.returncode, 0,
                             f"offline mode failed:\n{p.stderr}\n{p.stdout}")
            self.assertIn("summary: 17 passed, 0 failed", p.stderr)

            # Now run in online mode. The local_ok and
            # local_corrupt_download scenarios should also pass because
            # the mirror is reachable.
            p = self._run_cpp("--offline", "0", "--download", "1",
                              "--scenario", "all")
            self.assertEqual(p.returncode, 0,
                             f"online mode failed:\n{p.stderr}\n{p.stdout}")
            self.assertIn("summary: 17 passed, 0 failed", p.stderr)

    def test_end_to_end_download(self):
        """Demonstrate the full download path against a live
        serve_registry.py. We delete the model from model_dir and
        cache_dir, then call the C++ binary with --offline 0
        --download 1 against the live mirror; the binary should
        fetch the file from the server, verify its sha, and write
        it to both model_dir and cache_dir."""
        # Remove from both layers so ensure_model must download.
        (self.fx["model_dir"] / "my_test_model.mnn").unlink()
        (self.fx["cache_dir"] / "my_test_model.mnn").unlink(missing_ok=True)
        self.assertFalse((self.fx["model_dir"] / "my_test_model.mnn").exists())

        with _serve_registry(self.fx["mirror"], self.port):
            # Run a scenario that downloads into the supplied
            # model_dir. The "local_corrupt_download" scenario uses
            # a tmp model_dir (to be non-destructive), so we can't
            # reuse it. Instead, use the "local_ok" scenario to
            # verify the file is there. But local_ok SKIPs when
            # the file is missing. We need a different approach:
            # the C++ test's scenarios don't download into the
            # caller's model_dir. So we drive the end-to-end path
            # by writing a one-off C++ helper inline here.
            helper_src = self.workdir / "dl_helper.cpp"
            helper_src.write_text("""\
#include "ppocr/downloader.h"
#include <cstdio>
int main(int argc, char**argv) {
  if (argc < 5) return 2;
  ppocr::Registry r = ppocr::load_registry(argv[1]);
  for (const auto& e : r.entries) {
    auto res = ppocr::ensure_model(r, e.name, argv[2], argv[3], argv[4], 0, 1);
    std::fprintf(stderr, "ensure(%s) -> status=%d from_mirror=%d detail='%s'\\n",
                 e.name.c_str(), res.status, (int)res.from_mirror,
                 res.detail.c_str());
  }
  return 0;
}
""")
            helper_bin = self.workdir / "dl_helper"
            subprocess.check_call([
                "g++", "-std=c++17", "-O0",
                "-I", str(PPOCR_MNN_ROOT / "include"),
                "-I", str(PPOCR_MNN_ROOT / "third_party" / "MNN" / "include"),
                "-Wl,--whole-archive",
                str(BUILD_DIR / "libppocr_core.a"),
                "-Wl,--no-whole-archive",
                str(PPOCR_MNN_ROOT / "third_party" / "MNN" / "build" / "libMNN.a"),
                "-L/root/miniconda3/lib", "-lcurl",
                "-L/usr/lib/x86_64-linux-gnu", "-lz",
                "-lpthread", "-ldl",
                str(helper_src), "-o", str(helper_bin),
            ], stderr=subprocess.STDOUT)
            # Run the helper against the live server
            p = subprocess.run(
                [str(helper_bin),
                 str(self.fx["registry"]),
                 str(self.fx["model_dir"]),
                 str(self.fx["cache_dir"]),
                 f"http://127.0.0.1:{self.port}"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ,
                     "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu"})
            self.assertEqual(p.returncode, 0, f"helper failed:\n{p.stderr}")
            self.assertIn("status=0", p.stderr)
            self.assertIn("from_mirror=1", p.stderr)

            # After the helper, model_dir + cache_dir must have the file
            target_model = self.fx["model_dir"] / "my_test_model.mnn"
            target_cache = self.fx["cache_dir"] / "my_test_model.mnn"
            self.assertTrue(target_model.is_file(),
                            f"model file not created: {target_model}")
            self.assertTrue(target_cache.is_file(),
                            f"cache file not created: {target_cache}")
            # Sizes match
            self.assertEqual(target_model.stat().st_size, self.fx["size"])
            self.assertEqual(target_cache.stat().st_size, self.fx["size"])
            # SHAs match
            self.assertEqual(
                hashlib.sha256(target_model.read_bytes()).hexdigest(),
                self.fx["sha"])
            self.assertEqual(
                hashlib.sha256(target_cache.read_bytes()).hexdigest(),
                self.fx["sha"])
            # The server log must contain the GETs (helper binary
            # calls load_registry which reads registry.json, so
            # the GET /registry.json should be there).
            time.sleep(0.3)
            log = (self.fx["mirror"] / "serve.log").read_text()
            self.assertIn("GET /my_test_model.mnn", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
