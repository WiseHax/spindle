"""Serve the freshly built spindle.exe to the other PC on the LAN.

Run on the build machine (laptop):
    python tools/serve_exe.py              # serves target/release/spindle.exe on port 8765
    python tools/serve_exe.py --port 9000 --file path/to/spindle.exe

Only that one file is served (GET /spindle.exe, GET /spindle.exe.sha256), never the
directory. The script prints the exact PowerShell commands to run on the other PC.
Stop it with Ctrl+C once the copy is done.
"""

import argparse
import hashlib
import http.server
import os
import socket
import sys


def lan_ips():
    """IPv4 addresses of this machine that other LAN hosts can reach (best guess first)."""
    ips = set()
    try:
        # The OS picks the interface it would use to reach the LAN/internet; nothing is sent.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.1.1", 9))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips = [ip for ip in ips if not ip.startswith("127.")]
    # Home LANs first; virtual adapters (WSL 172.x, VirtualBox 192.168.56.x) last.
    return sorted(ips, key=lambda ip: (not ip.startswith("192.168.1."), ip.startswith("192.168.56."), ip.startswith("172."), ip))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=os.path.join(root, "target", "release", "spindle.exe"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="0.0.0.0", help="address to listen on (default: all interfaces)")
    args = ap.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}\nbuild it first:  cargo build --release")
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest().upper()
    name = os.path.basename(path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == f"/{name}":
                body, ctype = data, "application/octet-stream"
            elif self.path == f"/{name}.sha256":
                body, ctype = (digest + "\n").encode(), "text/plain"
            else:
                self.send_error(404, "only /%s is served" % name)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            print(f"  {self.client_address[0]}  {fmt % a}")

    ip = (lan_ips() or ["<this-pc-ip>"])[0]
    url = f"http://{ip}:{args.port}/{name}"
    print(f"Serving {path}")
    print(f"  {len(data) / 1e6:.2f} MB  sha256 {digest}")
    print(f"  URL: {url}")
    if len(lan_ips()) > 1:
        print(f"  (other addresses: {', '.join(lan_ips()[1:])})")
    print()
    print("On the other PC (PowerShell, in the folder with the .gguf), stop the running worker, then:")
    print()
    print(f"  Invoke-WebRequest {url} -OutFile spindle.new.exe")
    print(f"  if ((Get-FileHash spindle.new.exe).Hash -eq '{digest}') {{ Move-Item -Force spindle.new.exe spindle.exe; 'OK: spindle.exe updated' }} else {{ 'HASH MISMATCH - not installed' }}")
    print()
    print("Ctrl+C here when done.")
    try:
        http.server.ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
