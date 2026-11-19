#!/usr/bin/env python3
"""Evict the page cache of every file under the given directories (posix_fadvise DONTNEED), no sudo needed.
On the Orin Nano the GPU shares RAM with the page cache, and cached model files can make a 5 GB
allocation (the 2B Apollo models) fail although `free` shows enough. Usage: drop_file_cache.py DIR..."""
import os
import sys

n = 0
os.sync()
for top in sys.argv[1:]:
    for root, _, files in os.walk(os.path.expanduser(top)):
        for f in files:
            try:
                fd = os.open(os.path.join(root, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
print(f"evicted page cache of {n} files")
