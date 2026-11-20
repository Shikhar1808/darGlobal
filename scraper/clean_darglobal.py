"""Clean DarGlobal JSON exported from the browser (data/darglobal_*.json), in place.

- drops CMS layout noise (hex colours, 'paragraph', 'left', 2-3 letter codes)
- fills blank titles from the first text line
- merges records with identical text (the /one-of-one/* pages all serve one collection page)
"""
import json, re, sys
from pathlib import Path

NOISE = re.compile(r"^([0-9A-Fa-f]{6}|[A-Z]{2,3}|paragraph|text|left|right|null|VIEW ALL)$")


def clean(records):
    by_text = {}
    for r in records:
        lines = [l for l in r["text"].split("\n") if not NOISE.match(l.strip())]
        r["text"] = "\n".join(lines)
        if not (r.get("title") or "").strip() or "|" in r["title"]:  # '|' = generic <title> fallback, not the project name
            r["title"] = lines[0] if lines else r["url"].rsplit("/", 1)[-1]
        if r["text"] in by_text:
            by_text[r["text"]].setdefault("also_urls", []).append(r["url"])
        else:
            by_text[r["text"]] = r
    out = list(by_text.values())
    for r in out:
        if r.get("also_urls"):
            r["title"] = "One of One / Premiere Edition collection" if "/one-of-one/" in r["url"] else r["title"]
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        p = Path(path)
        recs = json.loads(p.read_text(encoding="utf-8"))
        out = clean(recs)
        assert all(r["title"] and r["text"] for r in out), "every record needs a title and text"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{p.name}: {len(recs)} -> {len(out)} records")
