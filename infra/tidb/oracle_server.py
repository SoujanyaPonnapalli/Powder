#!/usr/bin/env python3
"""Loopback-only, read-only placement advice for a patched PD.

Run with a precomputed per-deployment score file; a missing score makes PD use
its native safe selection. This process never sends PD operators.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from powder.deploy.pd_oracle import OracleError, ScoreTable  # noqa: E402

MAX_REQUEST_BYTES = 64 * 1024


def handler_for(table: ScoreTable) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/recommend":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > MAX_REQUEST_BYTES:
                    self.send_error(413)
                    return
                request = json.loads(self.rfile.read(length))
                response = table.recommend(request)
            except (ValueError, OracleError) as exc:
                self.send_error(400, str(exc))
                return
            payload = json.dumps(response, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="precomputed score table for this PD cluster")
    parser.add_argument("--port", type=int, default=8650)
    args = parser.parse_args()
    table = ScoreTable.load(args.scores)
    with HTTPServer(("127.0.0.1", args.port), handler_for(table)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
