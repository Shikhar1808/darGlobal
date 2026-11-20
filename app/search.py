"""Hybrid search over the scraped data, in memory.
Listings: exact filters, then ranked by meaning (embeddings) plus keyword overlap.
Projects: page text split into chunks; a project scores by its best chunk plus keyword overlap."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from app import currency

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LISTINGS = json.loads((DATA / "wasalt_listings.json").read_text(encoding="utf-8"))
PROJECTS = json.loads((DATA / "darglobal_projects.json").read_text(encoding="utf-8"))
COMPANY = json.loads((DATA / "darglobal_company.json").read_text(encoding="utf-8"))  # about, FAQ, visas, offices, partners

# common spellings -> substring of Wasalt's city value
CITY_ALIASES = {"dammam": "dammam", "taif": "tayif", "mecca": "makkah", "medina": "madinah",
                "dhahran": "zahran", "buraidah": "bariduh", "buraydah": "bariduh", "hail": "hayil", "ahsa": "ahsa"}
STOP = set("a an the in on at of for to and or is are i me my we you with any some do does have has show find "
           "what which where how much many there their this that from by about please want need looking".split())


EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 220 MB, Arabic + English in one space
KEYWORD_WEIGHT = 0.05  # one exact word match ~ +0.05 cosine, so names like "DG1" or "Trump" still win
MIN_PROJECT_SCORE = 0.35  # below this a project is unrelated to the question; don't send its text


def words(text):
    return {w for w in re.findall(r"[a-z0-9؀-ۿ]+", (text or "").lower()) if w not in STOP and len(w) > 1}


def _city_match(want, city):
    want = re.sub(r"^al[\s-]+", "", want.lower().strip()).replace("-", " ")  # only a leading "Al-"/"Al "
    want = CITY_ALIASES.get(want, want)
    city = (city or "").lower()
    return want in city or want.replace(" ", "") in city.replace(" ", "")


def rent_freq(listing):
    freqs = listing.get("rent_frequency") or {}
    for name, f in freqs.items():
        if f.get("default_freq"):
            return name
    return next(iter(freqs), None)


def filter_listings(f):
    """f: dict from the extract step. Unknown/None keys are ignored."""
    out = []
    for l in LISTINGS:
        if f.get("city") and not _city_match(f["city"], l["city"]):
            continue
        if f.get("purpose") and l["purpose"] != f["purpose"]:
            continue
        if f.get("property_type") and (l["sub_type"] or "").lower() != f["property_type"].lower():
            continue
        if f.get("bedrooms") and l["bedrooms"] != f["bedrooms"]:
            continue
        if f.get("min_bedrooms") and (l["bedrooms"] or 0) < f["min_bedrooms"]:
            continue
        if f.get("max_price") and yearly_price(l) > f["max_price"]:
            continue
        if f.get("min_price") and yearly_price(l) < f["min_price"]:
            continue
        out.append(l)
    return out


def yearly_price(l):
    """Budgets for rent are yearly (extract prompt), so monthly rents are compared x12."""
    price = l["price_sar"] or 0
    return price * 12 if l["purpose"] == "rent" and rent_freq(l) == "monthly" else price


def search_listings(f, question, k=8):
    """Filter, relaxing price then bedrooms if nothing matches. Returns (listings, relaxed_keys)."""
    relaxed = []
    hits = filter_listings(f)
    for keys in (("max_price", "min_price"), ("bedrooms", "min_bedrooms")):
        if hits:
            break
        present = [k_ for k_ in keys if f.get(k_)]  # only report filters the user actually set
        if present:
            f = {k_: v for k_, v in f.items() if k_ not in present}
            relaxed += present
            hits = filter_listings(f)
    keywords = f.get("keywords") or []
    q, qv = words(question) | words(" ".join(keywords)), query_vector(question, keywords)
    hits.sort(key=lambda l: -(float(LISTING_VECS[POS[l["url"]]] @ qv)
                              + KEYWORD_WEIGHT * len(q & words(_listing_text(l)))))
    return hits[:k], relaxed


def search_projects(question, keywords=(), k=3):
    q, qv = words(question) | words(" ".join(keywords)), query_vector(question, keywords)
    best = np.full(len(PROJECTS), -1.0)
    np.maximum.at(best, CHUNK_OWNER, CHUNK_VECS @ qv)  # each project's best-matching chunk

    def score(i):
        p = PROJECTS[i]
        return best[i] + KEYWORD_WEIGHT * (3 * len(q & words(p["title"])) + 2 * len(q & words(p.get("location")))
                                           + len(q & words(p["text"])))

    ranked = sorted(range(len(PROJECTS)), key=score, reverse=True)
    return [PROJECTS[i] for i in ranked[:k] if score(i) >= MIN_PROJECT_SCORE]


def search_company(question, keywords=(), k=3):
    """Company pages (one short record each) ranked the same way as projects."""
    q, qv = words(question) | words(" ".join(keywords)), query_vector(question, keywords)
    scores = COMPANY_VECS @ qv + KEYWORD_WEIGHT * np.array([len(q & words(c["title"] + " " + c["text"])) for c in COMPANY])
    return [COMPANY[i] for i in np.argsort(-scores)[:k] if scores[i] >= MIN_PROJECT_SCORE]


def company_block(c):
    return f"### {c['title']}\nURL: {c['url']}\n{c['text']}"


def _price(l, cur=None):
    """'340,000 SAR/yearly', plus '(about 68K GBP)' when the user thinks in another currency."""
    per = f"/{rent_freq(l)}" if l["purpose"] == "rent" and rent_freq(l) else ""
    out = f"{l['price_sar']:,} SAR{per}"
    if cur and cur != "SAR" and l["price_sar"]:
        out += f" (about {currency.fmt(currency.from_sar(l['price_sar'], cur))} {cur}{per})"
    return out


def listing_line(l, cur=None):
    beds = f"{l['bedrooms']} bed" if l["bedrooms"] else ""
    area = f"{l['area_sqm']} sqm" if l["area_sqm"] else ""
    return " | ".join(x for x in [l["title"], f"for {l['purpose']}", _price(l, cur), beds, area,
                                  l["furnishing"] or "", f"{l['district']}, {l['city']}", l["url"]] if x)


def listing_card(l, cur=None):
    """What the page shows as a result card (only for listings the answer mentions)."""
    return {"kind": "listing", "title": l["title"], "badge": f"For {l['purpose']}", "price": _price(l, cur),
            "facts": " · ".join(x for x in [f"{l['bedrooms']} bed" if l["bedrooms"] else "",
                                            f"{l['area_sqm']} sqm" if l["area_sqm"] else "", l["furnishing"] or ""] if x),
            "place": ", ".join(x for x in [l["district"], l["city"]] if x), "url": l["url"], "source": "Wasalt.sa"}


def _shorten(text, n):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def project_card(p):
    return {"kind": "project", "title": p["title"], "badge": "DarGlobal project", "price": p.get("starting_price") or "",
            "facts": _shorten(p.get("meta_description") or "", 140), "place": p.get("location") or "", "url": p["url"],
            "source": "DarGlobal"}


def project_block(p, chars=800):
    head = f"### {p['title']} ({p.get('location') or 'location n/a'})\nURL: {p['url']}"
    if p.get("starting_price"):
        head += f"\nStarting price: {p['starting_price']}"
    return f"{head}\n{p.get('meta_description') or ''}\n{p['text'][:chars]}"


PROJECT_INDEX = "\n".join(f"- {p['title']} ({p.get('location') or 'n/a'}): {p['url']}" for p in PROJECTS)


# ---- embeddings ----
def _listing_text(l):
    return f"{l['title']} {l['sub_type'] or ''} {l['district'] or ''} {l['city'] or ''} {l['description'] or ''}"


def _chunks(p, size=400):
    """Title + location prefix on every chunk so each chunk knows which project it belongs to."""
    head = f"{p['title']} ({p.get('location') or ''}): "
    body = f"{p.get('meta_description') or ''}\n{p['text']}"
    return [head + body[i:i + size] for i in range(0, len(body), size)]


_model = TextEmbedding(EMBED_MODEL, cache_dir=str(ROOT / ".models"))


def _embed(texts):
    v = np.array(list(_model.embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # unit length, so dot product = cosine similarity


def query_vector(question, keywords=()):
    return _embed([f"{question} {' '.join(keywords)}"])[0]


def _load_vectors():
    """Embeds all records once and saves them; recomputed only when the data or model changes."""
    listing_texts = [_listing_text(l)[:600] for l in LISTINGS]
    chunks = [(i, c) for i, p in enumerate(PROJECTS) for c in _chunks(p)]
    company_texts = [f"{c['title']}: {c['text']}" for c in COMPANY]
    key = hashlib.sha256(json.dumps([EMBED_MODEL, listing_texts, chunks, company_texts]).encode()).hexdigest()
    path = DATA / "embeddings.npz"
    if path.exists():
        saved = np.load(path)
        if str(saved["key"]) == key:
            return saved["listings"], saved["chunks"], saved["owner"], saved["company"]
    listing_vecs, chunk_vecs = _embed(listing_texts), _embed([c for _, c in chunks])
    owner = np.array([i for i, _ in chunks])
    company_vecs = _embed(company_texts)
    np.savez(path, key=key, listings=listing_vecs, chunks=chunk_vecs, owner=owner, company=company_vecs)
    return listing_vecs, chunk_vecs, owner, company_vecs


LISTING_VECS, CHUNK_VECS, CHUNK_OWNER, COMPANY_VECS = _load_vectors()
POS = {l["url"]: i for i, l in enumerate(LISTINGS)}
