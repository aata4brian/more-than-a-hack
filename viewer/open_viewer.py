"""Open the tactical video and predictions through a loopback-only viewer."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def settings():
    path = ROOT / ".mtah-local.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def resolve_assets(data_dir=None, prediction=None):
    config = settings()
    data = Path(data_dir or os.environ.get("MTAH_DATA_DIR") or config.get("data_dir") or ROOT / "data").expanduser().resolve()
    while not (data / "tactical" / "BAR-ATH_tactical.mp4").is_file():
        if data_dir or not sys.stdin.isatty():
            raise FileNotFoundError(f"Video tactical belum ditemukan di {data}")
        value = input("Folder yang berisi tactical/ dan broadcast/ (kosong untuk keluar): ").strip().strip('"')
        if not value:
            raise FileNotFoundError("Folder video belum dipilih.")
        data = Path(value).expanduser().resolve()
    chosen = prediction or config.get("prediction")
    if chosen:
        pred = Path(chosen).expanduser().resolve()
        if not pred.is_file():
            raise FileNotFoundError(f"CSV prediksi belum ditemukan: {pred}")
    else:
        candidates = list((ROOT / "outputs").glob("**/Hijrah ke Barcelona.csv"))
        pred = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return {
        "video": data / "tactical" / "BAR-ATH_tactical.mp4",
        "predictions.csv": pred,
        "reference.csv": ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv",
        "tracking.txt": data / "tactical" / "BAR-ATH_tactical_atd.txt",
    }


def make_handler(assets, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.serve(False)

        def do_GET(self):
            self.serve(True)

        def serve(self, body):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                self.send_error(403)
                return
            prefix = "/" + token + "/"
            path = self.path.split("?", 1)[0]
            if not path.startswith(prefix):
                self.send_error(404)
                return
            route = path[len(prefix):]
            if route == "manifest.json":
                payload = json.dumps({key: value.name for key, value in assets.items()
                                      if value is not None and value.is_file()}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if body:
                    self.wfile.write(payload)
                return
            source = ROOT / "viewer" / "Viewer_Hijrah_ke_Barcelona.html" if route == "" else assets.get(route)
            if source is None or not source.is_file():
                self.send_error(404)
                return
            total = source.stat().st_size
            start, end = 0, total - 1
            value = self.headers.get("Range")
            if value:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
                if not match or not any(match.groups()):
                    self.invalid_range(total)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), total - 1) if last else total - 1
                else:
                    start = max(0, total - int(last))
                if start >= total or end < start:
                    self.invalid_range(total)
                    return
            mime = "text/html; charset=utf-8" if route == "" else (
                "video/mp4" if route == "video" else "text/plain; charset=utf-8")
            try:
                with source.open("rb") as stream:
                    stream.seek(start)
                    self.send_response(206 if value else 200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    if value:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                    self.end_headers()
                    if body:
                        remaining = end - start + 1
                        while remaining:
                            chunk = stream.read(min(256 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except OSError:
                self.close_connection = True

        def invalid_range(self, total):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.send_header("Content-Length", "0")
            self.end_headers()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        assets = resolve_assets(args.data_dir, args.prediction)
        for key, path in assets.items():
            print(f"{key}: {path.name if path and path.is_file() else 'belum tersedia'}")
        if args.check:
            return 0
        token = secrets.token_urlsafe(24)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(assets, token))
        url = f"http://127.0.0.1:{server.server_port}/{token}/?autoload=1"
        print(f"Viewer: {url}", flush=True)
        print("Biarkan terminal terbuka. Ctrl+C untuk menutup viewer.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (OSError, ValueError) as error:
        print(f"Gagal: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
