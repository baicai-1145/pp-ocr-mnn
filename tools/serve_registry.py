#!/usr/bin/env python3
"""serve_registry.py — minimal local HTTP file server for the auto-downloader.

Per TOOLS-5 / M3: the runtime downloader needs a base URL to fetch
models from. In production this will point to GitHub Releases / HF /
self-hosted CDN; for local development and tests we ship a tiny
stdlib-only file server that exposes a directory of `.mnn` files plus
the matching `registry.json` over plain HTTP.

Endpoints:
  GET /<path>          -> file (404 if missing)
  GET /registry.json   -> the registry (must be in the served dir)
  GET /sha256/<name>   -> hex sha256 of a model (for tests that want to
                         confirm the server side independently)
  GET /healthz         -> "ok" (liveness)

The server is single-threaded by default (the C++ downloader uses
sequential requests per file). For multi-threaded testing, pass
--threaded which uses ThreadingHTTPServer.

Usage:
  serve_registry.py --root <dir> [--port 8765] [--bind 127.0.0.1]
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import os
import socketserver
import sys
import threading
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class RegistryHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files from `--root` and add a few diagnostic endpoints.

    `SimpleHTTPRequestHandler.translate_path` already does the
    path-join-with-root, so we just override a couple of routes."""

    root: Path = Path(".")

    def translate_path(self, path: str) -> str:
        # Strip query, normalize, then prepend our root. The default
        # implementation already does the latter half; we just override
        # the working-directory by prepending root.
        rel = super().translate_path(path)
        rel = rel.lstrip("/")
        if rel and not rel.startswith(str(self.root)):
            rel = str(self.root / rel)
        return rel

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path.startswith("/sha256/"):
            name = self.path[len("/sha256/"):]
            if "?" in name:
                name = name.split("?", 1)[0]
            target = self.root / name
            if not target.is_file():
                self.send_error(404, f"no such file: {name}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(sha256_of(target).encode("ascii"))
            return
        if self.path == "/registry.json":
            target = self.root / "registry.json"
            if not target.is_file():
                self.send_error(404, "registry.json not in --root")
                return
            return self._serve_file(target, "application/json")
        if self.path == "/" or self.path == "":
            # Index: list .mnn files
            names = sorted(p.name for p in self.root.glob("*.mnn"))
            body = ("\n".join(names) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # Default: serve any file from --root. We need to sanitize
        # the path to prevent directory-traversal.
        rel = self.path.lstrip("/")
        if "?" in rel:
            rel = rel.split("?", 1)[0]
        if ".." in rel.split("/"):
            self.send_error(403, "forbidden")
            return
        target = (self.root / rel).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            self.send_error(403, "forbidden")
            return
        if not target.is_file():
            self.send_error(404, f"no such file: {rel}")
            return
        return self._serve_file(target, "application/octet-stream")

    def _serve_file(self, path: Path, ctype: str):
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def log_message(self, fmt, *args):
        # One line per request, prefixed with [SERVE] for easy filtering.
        sys.stderr.write("[SERVE] " + (fmt % args) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="serve_registry",
                                 description="Local HTTP file server for "
                                             "testing the auto-downloader")
    ap.add_argument("--root", required=True, help="directory to serve from")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="interface to bind (default 127.0.0.1; use 0.0.0.0 for tests)")
    ap.add_argument("--threaded", action="store_true",
                    help="use ThreadingHTTPServer (one thread per request)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"serve_registry: --root {root} is not a directory", file=sys.stderr)
        return 2
    RegistryHandler.root = root

    cls = http.server.ThreadingHTTPServer if args.threaded else socketserver.TCPServer
    # Allow fast restart on the same port (avoids "Address already in use"
    # when a test harness restarts the server in quick succession).
    cls.allow_reuse_address = True
    srv = cls((args.bind, args.port), RegistryHandler)
    print(f"serve_registry: serving {root} on http://{args.bind}:{args.port}",
          file=sys.stderr)
    print(f"  endpoints: /registry.json, /<name>.mnn, /sha256/<name>, /healthz",
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
