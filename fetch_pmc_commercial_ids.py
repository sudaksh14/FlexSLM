import time, json, urllib.request, urllib.parse
from datetime import date, timedelta

BASE = "https://eutils.ncbi.nlm.nih.gov/eutils/esearch.fcgi"
LICENSE_FILTER = ("(cc0_license[filter] OR cc_by_license[filter] "
                   "OR cc_by-sa_license[filter] OR cc_by-nd_license[filter])")
TARGET = 2_000_000
OUT = "pmc_commercial_ids.txt"

def fetch_json(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            if "ERROR" in data.get("esearchresult", {}):
                raise RuntimeError(data["esearchresult"]["ERROR"])
            return data
        except Exception as e:
            print(f"retry after error: {e}", flush=True)
            time.sleep(2)
    raise RuntimeError("failed after 3 attempts")

def term_for_range(d1, d2):
    return f'open access[filter] AND {LICENSE_FILTER} AND ("{d1.isoformat()}"[PDAT] : "{d2.isoformat()}"[PDAT])'

ids = []

def collect(d1, d2):
    if len(ids) >= TARGET:
        return
    data = fetch_json({"db": "pmc", "term": term_for_range(d1, d2), "retmax": 0, "retmode": "json"})
    time.sleep(0.34)
    count = int(data["esearchresult"]["count"])
    if count == 0:
        return
    if count <= 9999:
        data = fetch_json({"db": "pmc", "term": term_for_range(d1, d2), "retmax": count, "retmode": "json"})
        time.sleep(0.34)
        batch = data["esearchresult"]["idlist"]
        ids.extend(batch)
        print(f"[{d1} - {d2}] +{len(batch)} ids, total={len(ids)}", flush=True)
        return
    mid = d1 + (d2 - d1) // 2
    if mid <= d1 or mid >= d2:
        data = fetch_json({"db": "pmc", "term": term_for_range(d1, d2), "retmax": 9999, "retmode": "json"})
        time.sleep(0.34)
        batch = data["esearchresult"]["idlist"]
        ids.extend(batch)
        print(f"[{d1} - {d2}] (unsplittable) +{len(batch)} ids, total={len(ids)}", flush=True)
        return
    collect(d1, mid)
    collect(mid + timedelta(days=1), d2)

collect(date(1990, 1, 1), date(2026, 12, 31))

with open(OUT, "w") as f:
    for i in ids[:TARGET]:
        f.write(f"PMC{i}\n")

print(f"DONE: wrote {min(len(ids), TARGET)} ids to {OUT}")
