"""
Download files from a public SURFdrive (Nextcloud) share over WebDAV, resumable.
    python scripts/fetch_surfdrive.py TOKEN OUT_DIR PATH [PATH ...] [--exclude SUBSTR ...]
Each PATH is a file or a directory inside the share (e.g. repo, saved_models/x.pt); directories
are walked recursively. Files whose local size already equals the remote size are skipped.
"""
import argparse
import base64
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "https://surfdrive.surf.nl/public.php/webdav/"


def request(url, token, method="GET", headers=None):
    auth = base64.b64encode(f"{token}:".encode()).decode()
    h = {"Authorization": f"Basic {auth}", **(headers or {})}
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers=h), timeout=120)


def listing(token, path):
    body = request(BASE + urllib.parse.quote(path), token, "PROPFIND", {"Depth": "1"}).read().decode()
    out = []
    for m in re.finditer(r"<d:response>(.*?)</d:response>", body, re.S):
        b = m.group(1)
        href = urllib.parse.unquote(re.search(r"<d:href>(.*?)</d:href>", b).group(1))
        rel = href.split("/public.php/webdav/")[-1]
        size = re.search(r"getcontentlength>(\d+)", b)
        out.append((rel, int(size.group(1)) if size else None, "<d:collection" in b))
    return out


def walk(token, path, exclude):
    entries = listing(token, path)
    if len(entries) == 1 and not entries[0][2]:  # a single file
        yield entries[0][0], entries[0][1]
        return
    for rel, size, is_dir in entries:
        if rel.rstrip("/") == path.rstrip("/") or any(x in rel for x in exclude):
            continue
        if is_dir:
            yield from walk(token, rel.rstrip("/"), exclude)
        else:
            yield rel, size


def download(token, rel, size, out_dir):
    dest = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size is not None and have == size:
        return
    mode, headers = "wb", {}
    if 0 < have < (size or 0):
        mode, headers = "ab", {"Range": f"bytes={have}-"}
    with request(BASE + urllib.parse.quote(rel), token, "GET", headers) as r, open(dest, mode) as f:
        while chunk := r.read(8 << 20):
            f.write(chunk)
    print(f"  {rel}  {os.path.getsize(dest) / 1e9:.2f} GB", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("token")
    p.add_argument("out_dir")
    p.add_argument("paths", nargs="+")
    p.add_argument("--exclude", nargs="*", default=[])
    a = p.parse_args()
    for path in a.paths:
        for rel, size in walk(a.token, path.rstrip("/"), a.exclude):
            download(a.token, rel, size, a.out_dir)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
