#!/usr/bin/env python3
"""tests/test_downloader.py — Python driver for the C++ auto-downloader (TOOLS-5).

Drives the real C++ `test_downloader` binary against a live
`tools/serve_registry.py` HTTP mirror, asserting the 4 scenarios
required by the TOOLS-5 brief:

  1. 正常下载 (normal download)
  2. sha 不匹配 (sha mismatch -> ERR_DOWNLOAD, no file on disk)
  3. 离线缺文件 (offline=1 + missing -> ERR_MODEL)
  4. cache 命中 (cache hit -> from_cache=1, no re-download)

Each scenario spins up a fresh serve_registry, builds a 10 KB
random-byte "model" + matching registry, and runs the C++ binary
as a subprocess. Tests are self-contained; no fixtures outside
`/tmp` are touched.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
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
# The worktree's source root (where downloader.h, downloader.cpp live).
SRC_ROOT = Path(os.environ.get("PPOCR_SRC_ROOT", REPO_ROOT))
# pp-ocr-mnn source tree (sibling of the worktree, for MNN includes).
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
        start_new_session=True,
    )
    # Wait for the port to accept connections (up to 5s)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if _port_free(port):
            time.sleep(0.1)
            continue
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
    """Create a small registry + one .mnn under `root / mirror`."""
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
        "data": random_data,
    }


def _write_wrong_sha_registry(reg_path: Path, expected_sha: str):
    """Mutate the registry on disk to advertise a wrong sha so the
    next download is rejected by verify_sha."""
    r = json.loads(reg_path.read_text())
    # Replace the first 8 hex chars of every entry's sha with 0s.
    for v in r.values():
        v["sha256"] = "0" * 8 + v["sha256"][8:]
    reg_path.write_text(json.dumps(r, indent=2))


def _build_cpp_helper(out_path: Path) -> Path:
    """Compile a small C++ helper that calls ppocr::ensure_model."""
    src = out_path.parent / "dl_helper.cpp"
    src.write_text("""\
#include "ppocr/downloader.h"
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
  if (argc < 5) return 2;
  int offline = (argc > 5) ? std::atoi(argv[5]) : 0;
  int download = (argc > 6) ? std::atoi(argv[6]) : 1;
  ppocr::Registry r = ppocr::load_registry(argv[1]);
  int last_status = 0;
  for (const auto& e : r.entries) {
    auto res = ppocr::ensure_model(r, e.name, argv[2], argv[3], argv[4],
                                    offline, download);
    std::fprintf(stderr, "ensure(%s) -> status=%d (%s) from_cache=%d "
                          "from_mirror=%d already_ok=%d detail='%s'\\n",
                 e.name.c_str(), res.status,
                 ppocr_status_string(res.status),
                 (int)res.from_cache, (int)res.from_mirror,
                 (int)res.already_ok, res.detail.c_str());
    last_status = res.status;
  }
  return last_status;
}
""")
    cmd = [
        "g++", "-std=c++17", "-O0",
        "-I", str(SRC_ROOT / "include"),
        "-I", str(PPOCR_MNN_ROOT / "third_party" / "MNN" / "include"),
        "-Wl,--whole-archive",
        str(BUILD_DIR / "libppocr_core.a"),
        "-Wl,--no-whole-archive",
        str(PPOCR_MNN_ROOT / "third_party" / "MNN" / "build" / "libMNN.a"),
        "-L/root/miniconda3/lib", "-lcurl",
        "-L/usr/lib/x86_64-linux-gnu", "-lz",
        "-lpthread", "-ldl",
        str(src), "-o", str(out_path),
    ]
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)
    return out_path


def _ensure_cpp_helper(workdir: Path) -> Path:
    """Compile the helper binary once per workdir and return its path."""
    helper = workdir / "dl_helper"
    if not helper.exists():
        _build_cpp_helper(helper)
    return helper


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
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/healthz") as r:
                    self.assertEqual(r.read().decode(), "ok")
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/x.mnn") as r:
                    self.assertEqual(r.read(), b"hello")
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/sha256/x.mnn") as r:
                    self.assertEqual(r.read().decode(),
                                     hashlib.sha256(b"hello").hexdigest())
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/") as r:
                    self.assertIn(b"x.mnn", r.read())
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/missing.mnn") as r:
                        self.assertEqual(r.status, 404)
                except urllib.error.HTTPError as e:
                    self.assertEqual(e.code, 404)


class TestDownloader(unittest.TestCase):
    """End-to-end downloader test (TOOLS-5 / M3) — 4 scenarios."""

    @classmethod
    def setUpClass(cls):
        if not TEST_BIN.exists():
            raise unittest.SkipTest(
                f"test_downloader not built at {TEST_BIN} (set "
                f"PPOCR_BUILD_DIR or run cmake --build first)")
        # Compile the helper binary once.
        cls._workdir = Path(tempfile.mkdtemp(prefix="dltest_helper_"))
        try:
            cls._helper = _build_cpp_helper(cls._workdir / "dl_helper")
        except Exception as ex:
            raise unittest.SkipTest(
                f"failed to build C++ helper: {ex}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._workdir, ignore_errors=True)

    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp(prefix="dltest_"))
        self.fx = _make_fixture(self.workdir)
        self.port = _allocate_port()

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _run_helper(self, offline: int = 0, download: int = 1
                     ) -> subprocess.CompletedProcess:
        cmd = [str(self._helper),
               str(self.fx["registry"]),
               str(self.fx["model_dir"]),
               str(self.fx["cache_dir"]),
               f"http://127.0.0.1:{self.port}",
               str(offline), str(download)]
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30, env={**os.environ,
                                              "LD_LIBRARY_PATH":
                                              "/usr/lib/x86_64-linux-gnu"})

    def _run_cpp_test(self, *args) -> subprocess.CompletedProcess:
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

    # ------------------------------------------------------------------
    # Scenarios required by the TOOLS-5 brief
    # ------------------------------------------------------------------

    def test_scenario_1_normal_download(self):
        """正常下载: clear model_dir, ensure_model must download."""
        # No file in model_dir, no file in cache_dir.
        self.assertFalse((self.fx["model_dir"] / "my_test_model.mnn").exists())
        self.assertFalse((self.fx["cache_dir"] / "my_test_model.mnn").exists())

        with _serve_registry(self.fx["mirror"], self.port):
            p = self._run_helper(offline=0, download=1)
            self.assertEqual(p.returncode, 0,
                             f"helper failed:\n{p.stderr}")
            self.assertIn("status=0", p.stderr)
            self.assertIn("from_mirror=1", p.stderr)

            # Files exist on disk in BOTH directories, with correct sha+size
            for d in [self.fx["model_dir"], self.fx["cache_dir"]]:
                f = d / "my_test_model.mnn"
                self.assertTrue(f.is_file(), f"missing: {f}")
                self.assertEqual(f.stat().st_size, self.fx["size"])
                self.assertEqual(
                    hashlib.sha256(f.read_bytes()).hexdigest(),
                    self.fx["sha"])

            # The server log shows the GET
            time.sleep(0.2)
            log = (self.fx["mirror"] / "serve.log").read_text()
            self.assertIn("GET /my_test_model.mnn", log)

    def test_scenario_2_sha_mismatch(self):
        """sha 不匹配: registry advertises wrong sha, download must
        be rejected and no file must be left on disk."""
        # Mutate the registry to advertise a wrong sha
        _write_wrong_sha_registry(self.fx["registry"], self.fx["sha"])

        with _serve_registry(self.fx["mirror"], self.port):
            p = self._run_helper(offline=0, download=1)
            # Return code is non-zero (ERR_DOWNLOAD = 4)
            self.assertNotEqual(p.returncode, 0,
                                f"helper should fail:\n{p.stderr}")
            self.assertIn("status=4", p.stderr,
                          f"expected status=4 (ERR_DOWNLOAD):\n{p.stderr}")
            self.assertIn("sha256 mismatch", p.stderr,
                          f"expected mismatch message:\n{p.stderr}")

            # No file in model_dir, no file in cache_dir
            self.assertFalse(
                (self.fx["model_dir"] / "my_test_model.mnn").exists())
            self.assertFalse(
                (self.fx["cache_dir"] / "my_test_model.mnn").exists())
            # The .part file is also cleaned up
            self.assertFalse(
                (self.fx["cache_dir"] / "my_test_model.mnn.part").exists())

    def test_scenario_3_offline_missing(self):
        """离线缺文件: offline=1 + file missing -> ERR_MODEL."""
        # No file anywhere
        self.assertFalse((self.fx["model_dir"] / "my_test_model.mnn").exists())
        self.assertFalse((self.fx["cache_dir"] / "my_test_model.mnn").exists())

        with _serve_registry(self.fx["mirror"], self.port):
            p = self._run_helper(offline=1, download=0)
            self.assertEqual(p.returncode, 2,
                             f"expected ERR_MODEL (2):\n{p.stderr}")
            self.assertIn("status=2", p.stderr)
            self.assertIn("offline=1", p.stderr)
            self.assertIn("refusing to download", p.stderr)
            # No file was downloaded
            self.assertFalse(
                (self.fx["model_dir"] / "my_test_model.mnn").exists())
            self.assertFalse(
                (self.fx["cache_dir"] / "my_test_model.mnn").exists())

    def test_scenario_4_cache_hit(self):
        """cache 命中: second call after download must reuse cache, not
        re-download (from_cache=1, no second GET to the server)."""
        with _serve_registry(self.fx["mirror"], self.port):
            # First call: download
            p1 = self._run_helper(offline=0, download=1)
            self.assertEqual(p1.returncode, 0, p1.stderr)
            self.assertIn("from_mirror=1", p1.stderr)
            # Capture the server log size to detect the second GET
            time.sleep(0.1)
            log_after_first = (self.fx["mirror"] / "serve.log").read_text()
            get_count_1 = log_after_first.count("GET /my_test_model.mnn")
            self.assertGreaterEqual(get_count_1, 1,
                                    "expected at least one GET in first run")

            # Now delete from model_dir; cache still has it
            (self.fx["model_dir"] / "my_test_model.mnn").unlink()
            self.assertFalse(
                (self.fx["model_dir"] / "my_test_model.mnn").exists())
            self.assertTrue(
                (self.fx["cache_dir"] / "my_test_model.mnn").exists())

            # Second call: should hit cache, not download
            p2 = self._run_helper(offline=0, download=1)
            self.assertEqual(p2.returncode, 0, p2.stderr)
            self.assertIn("from_cache=1", p2.stderr)
            self.assertIn("from_mirror=0", p2.stderr)

            # The server must NOT have received a second GET
            time.sleep(0.1)
            log_after_second = (self.fx["mirror"] / "serve.log").read_text()
            get_count_2 = log_after_second.count("GET /my_test_model.mnn")
            self.assertEqual(get_count_2, get_count_1,
                             f"server got extra GETs: "
                             f"first={get_count_1}, second={get_count_2}")

            # model_dir was repopulated from the cache
            self.assertTrue(
                (self.fx["model_dir"] / "my_test_model.mnn").is_file())
            self.assertEqual(
                (self.fx["model_dir"] / "my_test_model.mnn").stat().st_size,
                self.fx["size"])

    # ------------------------------------------------------------------
    # Additional coverage: C++ test driver's own scenario battery
    # ------------------------------------------------------------------

    def test_cpp_test_all_scenarios(self):
        """The bundled tests/test_downloader.cpp driver (17 scenario
        checks) must pass against the live mirror, in both offline
        and online mode."""
        with _serve_registry(self.fx["mirror"], self.port):
            # Place a valid copy in model_dir so local_ok has a target
            shutil.copy(self.fx["mirror"] / "my_test_model.mnn",
                        self.fx["model_dir"])
            p = self._run_cpp_test("--offline", "1", "--download", "0",
                                   "--scenario", "all")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("summary: 17 passed, 0 failed", p.stderr)

            p = self._run_cpp_test("--offline", "0", "--download", "1",
                                   "--scenario", "all")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("summary: 17 passed, 0 failed", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
