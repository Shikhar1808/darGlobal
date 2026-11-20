"""Evaluation of the chatbot on eval/cases.json.

  uv run python -m eval.run              # retrieval only: no model calls, compares keyword vs hybrid search
  uv run --env-file .env python -m eval.run --e2e   # also runs the real bot (2 model calls per case)

Writes eval/results.md.
"""
import json
import re
import sys
import time
from pathlib import Path

from langchain_core.caches import InMemoryCache

from app import graph, search

HERE = Path(__file__).parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
ARABIC = re.compile(r"[؀-ۿ]")


def keyword_baseline(question, k=3):
    """The search before vectors (D-007): 3 x title + 2 x location + 1 x body word overlap."""
    q = search.words(question)

    def score(p):
        return 3 * len(q & search.words(p["title"])) + 2 * len(q & search.words(p.get("location"))) + len(q & search.words(p["text"]))
    ranked = sorted(search.PROJECTS, key=score, reverse=True)
    return [p for p in ranked[:k] if score(p) > 0]


def hit(projects, expected):
    return any(e.lower() in p["title"].lower() for p in projects for e in expected)


def retrieval():
    rows, totals = [], {"keyword": 0, "hybrid": 0}
    cases = [c for c in CASES if c.get("projects")]
    for c in cases:
        kw, hy = hit(keyword_baseline(c["q"]), c["projects"]), hit(search.search_projects(c["q"]), c["projects"])
        totals["keyword"] += kw
        totals["hybrid"] += hy
        rows.append(f"| {c['q']} | {'✅' if kw else '❌'} | {'✅' if hy else '❌'} |")
    n = len(cases)
    summary = (f"| **Project hit@3** | **{totals['keyword']}/{n} ({totals['keyword'] / n:.0%})** "
               f"| **{totals['hybrid']}/{n} ({totals['hybrid'] / n:.0%})** |")
    company = [c for c in CASES if c.get("company")]
    found = [hit(search.search_company(c["q"], k=2), c["company"]) for c in company]
    totals["company"], totals["company_n"] = sum(found), len(company)
    company_rows = "\n".join(f"| {c['q']} | {'✅' if ok else '❌'} |" for c, ok in zip(company, found))
    return ("## Retrieval: is an expected project in the top 3?\n\nNo model calls. Keyword = the search before vectors.\n\n"
            "| Question | Keyword | Hybrid |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n" + summary
            + "\n\n## Retrieval: is the right company page (about, FAQ, visas, offices) in the top 2?\n\n"
            f"| Question | Hybrid |\n| --- | --- |\n{company_rows}\n| **Company page hit@2** | **{sum(found)}/{len(company)}** |"), totals, n


def same(want, got):
    if isinstance(want, int):
        return isinstance(got, int) and abs(got - want) <= 0.01 * want
    return isinstance(got, str) and want.lower() in got.lower()


def e2e():
    graph.cache = InMemoryCache()  # measure the model, not the cache
    rows, lat, errors = [], [], 0
    score = {"source": 0, "fields": 0, "fields_total": 0, "projects": 0, "projects_total": 0, "lang": 0, "attrib": 0,
             "attrib_total": 0, "removed": 0}
    for i, c in enumerate(CASES):
        t, thread = time.time(), f"eval-{time.time_ns()}-{i}"
        try:
            reply = graph.chat(c["q"], thread)
        except Exception as e:  # rate limit etc.
            errors += 1
            rows.append(f"| {c['q']} | error: {type(e).__name__} | | | |")
            continue
        lat.append(time.time() - t)
        state = graph.graph.get_state({"configurable": {"thread_id": thread}}).values
        f, ctx = state["filters"], state.get("context") or ""
        ok_source = f["source"] == c["source"] or (c["source"] != "none" and f["source"] == "both")
        score["source"] += ok_source
        wrong = [k for k, v in (c.get("filters") or {}).items() if not same(v, f.get(k))]
        score["fields"] += len(c.get("filters") or {}) - len(wrong)
        score["fields_total"] += len(c.get("filters") or {})
        if c.get("projects") or c.get("company"):
            found = any(e.lower() in ctx.lower() for e in c.get("projects") or c["company"])
            score["projects"] += found
            score["projects_total"] += 1
        score["lang"] += bool(ARABIC.search(c["q"])) == bool(ARABIC.search(reply))
        if "## Wasalt listings" in ctx and "No listings match" not in ctx:
            score["attrib_total"] += 1
            score["attrib"] += "Source: Wasalt.sa" in reply
        removed = reply.count("(link removed)")
        score["removed"] += removed
        rows.append(f"| {c['q']} | {f['source']}{'' if ok_source else ' ❌'} | {', '.join(wrong) or '✅'} "
                    f"| {removed} | {lat[-1]:.1f}s |")
    n = len(CASES) - errors
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else 0  # noqa: E731
    summary = "\n".join([
        "| Metric | Result |", "| --- | --- |",
        f"| Source routing (darglobal / wasalt / none) | {score['source']}/{n} ({score['source'] / max(n, 1):.0%}) |",
        f"| Filter fields extracted correctly | {score['fields']}/{score['fields_total']} ({score['fields'] / max(score['fields_total'], 1):.0%}) |",
        f"| Expected project or company page reached the model | {score['projects']}/{score['projects_total']} |",
        f"| Reply in the question's language | {score['lang']}/{n} |",
        f"| \"Source: Wasalt.sa\" when listings used | {score['attrib']}/{score['attrib_total']} |",
        f"| Invented links removed by the guard | {score['removed']} |",
        f"| Latency p50 / p95 | {p(0.5):.1f}s / {p(0.95):.1f}s |",
        f"| Errors (rate limits, timeouts) | {errors}/{len(CASES)} |",
    ])
    return (f"## End to end (model: `{graph.MODEL}`)\n\n{summary}\n\n| Question | Source | Wrong fields | Links removed | Time |\n"
            "| --- | --- | --- | --- | --- |\n" + "\n".join(rows))


if __name__ == "__main__":
    report, totals, n = retrieval()
    print(f"retrieval hit@3: keyword {totals['keyword']}/{n}, hybrid {totals['hybrid']}/{n}; company page hit@2 {totals['company']}/{totals['company_n']}")
    parts = [f"# Evaluation results\n\n{len(CASES)} cases in `eval/cases.json` (English and Arabic). "
             f"Run on {time.strftime('%Y-%m-%d')}.", report]
    if "--e2e" in sys.argv:
        parts.append(e2e())
    (HERE / "results.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print("wrote eval/results.md")
