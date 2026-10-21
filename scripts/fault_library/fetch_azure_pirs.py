"""Fetch every Azure Post Incident Review (PIR) from the public status history.

    python scripts/fault_library/fetch_azure_pirs.py

The history page renders only the newest PIRs; older ones come from the
page's own paging endpoint (/en-us/statushistoryapi/?page=N&...). Every page is
saved under data/fault_library/raw/azure_pir/<date>/page-NNN.html, then parsed
into pirs.json: one record per PIR with tracking ID, date, title and full text.
PIRs are retained for 5 years, so re-run periodically to keep older ones.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://azure.status.microsoft"
FIRST = BASE + "/en-us/status/history/"
PAGE = BASE + "/en-us/statushistoryapi/?page={n}&serviceSlug=all&regionSlug={region}&startDate=all&shdate=all"
#: Azure US regions (status-page slugs). Paging per region as well as "all" cross-checks completeness.
US_REGION_SLUGS = ["us-east", "us-east-2", "us-central", "us-north-central", "us-south-central",
                   "us-west-central", "us-west", "us-west-2", "us-west-3"]


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "powder-fault-library (research)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def text_of(fragment: str) -> str:
    t = re.sub(r"<(script|style).*?</\1>", " ", fragment, flags=re.S)
    t = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n") if line.strip())


def parse(pages: list[str]) -> list[dict]:
    """Split pages into PIRs at each 'Tracking ID:' and recover month/day headers."""
    out = {}
    for page in pages:
        txt = text_of(page)
        lines = txt.split("\n")
        month = None
        idx = [i for i, l in enumerate(lines) if re.match(r"Tracking ID:\s*\S+", l)]
        for j, i in enumerate(idx):
            tid = re.match(r"Tracking ID:\s*(\S+)", lines[i]).group(1)
            # header lines just above: "<day>", "<MM/DD/YYYY>", "<title>"
            head = lines[max(0, i - 4):i]
            dt = next((h for h in head if re.fullmatch(r"\d{2}/\d{2}/\d{4}", h)), None)
            title = head[-1] if head else ""
            end = idx[j + 1] - 4 if j + 1 < len(idx) else len(lines)
            body = "\n".join(lines[i + 1:end])
            if tid not in out:
                out[tid] = {"tracking_id": tid, "date": dt, "title": title, "text": body}
    return sorted(out.values(), key=lambda r: (r["date"] or "", r["tracking_id"]))


def main() -> int:
    day = date.today().isoformat()
    outdir = ROOT / "data/fault_library/raw/azure_pir" / day
    outdir.mkdir(parents=True, exist_ok=True)
    pages = [get(FIRST)]
    (outdir / "landing.html").write_text(pages[0])
    regions_of: dict[str, set] = {}
    for region in ["all"] + US_REGION_SLUGS:
        n = 1
        while True:
            p = get(PAGE.format(n=n, region=region))
            if "Tracking ID" not in p:
                break
            pages.append(p)
            (outdir / f"{region}-page-{n:03d}.html").write_text(p)
            if region != "all":
                for tid in re.findall(r"Tracking ID:\s*([A-Z0-9_\-]+)", text_of(p)):
                    regions_of.setdefault(tid, set()).add(region)
            n += 1
            time.sleep(0.3)
        print(f"{region}: {n - 1} pages", flush=True)
    pirs = parse(pages)
    for p in pirs:
        p["status_page_us_regions"] = sorted(regions_of.get(p["tracking_id"], []))
    (outdir / "pirs.json").write_text(json.dumps({"fetched": day, "source": FIRST, "pirs": pirs}, indent=1))
    print(f"{len(pirs)} PIRs -> {outdir / 'pirs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
