"""Scrape a sample of public Wasalt listings into data/wasalt_listings.json.

Listing URLs come from the public sitemap; each page embeds its data in Next.js __NEXT_DATA__ JSON.
Contact fields (phone, WhatsApp, agent names) are deliberately not stored, per wasalt.sa/llms.txt.
"""
import gzip, html, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130 Safari/537.36"
SITEMAP = "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz"
OUT = Path(__file__).resolve().parent.parent / "data" / "wasalt_listings.json"
N_SALE, N_RENT = int(sys.argv[1]) if len(sys.argv) > 1 else 400, int(sys.argv[2]) if len(sys.argv) > 2 else 200
NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body


def spread(urls, n):
    # even stride over the sitemap instead of the first n, to cover many cities and types
    step = max(1, len(urls) // n)
    return urls[::step][:n]


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def parse(url, page):
    m = NEXT_DATA.search(page)
    d = json.loads(m.group(1))["props"]["pageProps"].get("propertyDetailsV3") if m else None
    if not d:
        return None
    pi = d["propertyInfo"]
    attrs = {a["key"]: a.get("value") for a in d.get("attributes") or []}
    extra = {a["key"]: a.get("value") for a in d.get("additionalAttributes") or [] if a.get("value") not in (None, "", "-")}
    return {
        "source": "Wasalt.sa",
        "url": url,
        "id": d["id"],
        "title": pi.get("title"),
        "purpose": pi.get("propertyFor"),  # sale | rent
        "price_sar": pi.get("salePrice") if pi.get("propertyFor") == "sale" else pi.get("expectedRent"),
        "rent_frequency": pi.get("rentFreq") or None,
        "main_type": pi.get("propertyMainType"),
        "sub_type": pi.get("propertySubType"),
        "bedrooms": attrs.get("noOfBedrooms"),
        "bathrooms": attrs.get("noOfBathrooms"),
        "area_sqm": attrs.get("builtUpArea") or attrs.get("landArea"),
        "furnishing": pi.get("furnishingType"),
        "facing": pi.get("facingType"),
        "city": pi.get("city"),
        "district": pi.get("district"),
        "territory": pi.get("territory"),
        "address": pi.get("address"),
        "lat": (d.get("location") or {}).get("lat"),
        "lon": (d.get("location") or {}).get("lon"),
        "completion": extra.get("completionYear"),
        "facade": extra.get("propertyFacade"),
        "project_id": pi.get("projectId"),
        "rega_verified": d.get("isRegaProp"),
        "published_at": d.get("publishedAt"),
        "updated_at": d.get("updatedAt"),
        "description": clean(pi.get("description")),
    }


def scrape(url):
    try:
        rec = parse(url, get(url).decode("utf-8", "replace"))
    except Exception as e:  # one bad page must not kill the run
        print("skip", url, e, file=sys.stderr)
        rec = None
    time.sleep(1)  # ~2 req/s total with 2 workers: "moderate" per llms.txt
    return rec


def main():
    urls = re.findall(r"<loc>([^<]+)</loc>", get(SITEMAP).decode())
    sale = [u for u in urls if "/property/sale/" in u]
    rent = [u for u in urls if "/property/rent/" in u]
    print(f"sitemap: {len(sale)} sale, {len(rent)} rent")
    targets = spread(sale, N_SALE) + spread(rent, N_RENT)
    with ThreadPoolExecutor(2) as ex:
        recs = [r for r in ex.map(scrape, targets) if r]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(recs)}/{len(targets)} -> {OUT}")


if __name__ == "__main__":
    main()
