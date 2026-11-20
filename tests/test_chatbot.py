import json
import re

from langchain_core.messages import AIMessage

from app import currency, graph, search


def test_listing_filters_are_exact():
    hits, relaxed = search.search_listings(
        {"city": "Jeddah", "purpose": "sale", "property_type": "Apartment", "bedrooms": 3, "max_price": 1_000_000}, "")
    assert hits and not relaxed
    assert all(l["city"] == "Jeddah" and l["purpose"] == "sale" and l["sub_type"] == "Apartment"
               and l["bedrooms"] == 3 and l["price_sar"] <= 1_000_000 for l in hits)


def test_city_alias_matches_wasalt_spelling():
    assert all(l["city"] == "Aldammam" for l in search.filter_listings({"city": "Dammam"}))
    for spelling, n in [("Dammam", 18), ("Al-Khobar", 18), ("Al Khobar", 18), ("Makkah Al Mukarramah", 9), ("Makkah", 9)]:
        assert len(search.filter_listings({"city": spelling})) == n, spelling


def test_impossible_price_is_relaxed_not_empty():
    hits, relaxed = search.search_listings({"city": "Riyadh", "purpose": "sale", "max_price": 1}, "")
    assert hits and relaxed == ["max_price"]  # never reports filters the user did not set


def test_project_keyword_search():
    titles = [p["title"] for p in search.search_projects("trump villas in oman")]
    assert any("Trump" in t and "Villas" in t for t in titles)
    assert search.search_projects("tell me about DG1")[0]["title"] == "DG1"


def test_monthly_rent_is_compared_as_yearly():
    monthly = [l for l in search.LISTINGS if l["purpose"] == "rent" and search.rent_freq(l) == "monthly"]
    assert monthly
    for l in monthly:  # a yearly budget equal to 12 x the monthly rent keeps it; just below drops it
        assert l in search.filter_listings({"max_price": l["price_sar"] * 12})
        assert l not in search.filter_listings({"max_price": l["price_sar"] * 12 - 1})


def test_rent_line_has_frequency():
    rent = next(l for l in search.LISTINGS if l["purpose"] == "rent" and l["rent_frequency"])
    assert "SAR/" in search.listing_line(rent)


def test_parse_json_tolerates_model_noise():
    f = graph._parse_json('Sure!\n```json\n{"source":"wasalt","city":"Riyadh","max_price":"1.2e6","bedrooms":"x"}\n```')
    assert f["source"] == "wasalt" and f["max_price"] == 1_200_000 and f["bedrooms"] is None
    assert graph._parse_json("not json")["source"] == "both"


def test_parse_json_odd_types_do_not_crash_search():
    f = graph._parse_json('{"source":"wasalt","city":["Riyadh","Jeddah"],"purpose":"Sale","property_type":" "}')
    assert f["city"] is None and f["purpose"] == "sale" and f["property_type"] is None
    assert search.search_listings(f, "x")[0]


class FakeLLM:
    """Stands in for OpenRouter: first call = extract JSON, second = answer (echoes the context it got)."""
    filters = '{"source":"wasalt","city":"Jeddah","purpose":"sale","bedrooms":3}'

    def invoke(self, msgs):
        system = msgs[0].content
        if system.startswith("Turn the user's"):
            return AIMessage(self.filters)
        return AIMessage(system)

    def stream(self, msgs):
        text = self.invoke(msgs).content
        for i in range(0, len(text), 50):
            yield AIMessage(text[i:i + 50])


def test_graph_end_to_end_with_memory(monkeypatch):
    monkeypatch.setattr(graph, "llm", FakeLLM())
    reply = graph.chat("3 bed apartments for sale in Jeddah", "t1")
    assert "Wasalt listings" in reply and "Jeddah" in reply
    assert "About DarGlobal" not in reply  # company pages only for DarGlobal questions (tokens)
    state = graph.graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert len(state["messages"]) == 2  # user + assistant kept for follow-ups


def test_api_streams_reply_and_returns_thread(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setattr(graph, "llm", FakeLLM())
    r = TestClient(main.app).post("/api/chat", json={"message": "3 bed for sale in Jeddah", "thread_id": "t2"})
    assert r.status_code == 200 and r.headers["x-thread-id"] == "t2"
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[0]["cards"] and all(c["url"].startswith("https://wasalt.sa/") for c in events[0]["cards"])
    assert "Jeddah" in "".join(e.get("t", "") for e in events)  # the streamed pieces join into the full reply


def test_api_model_failure_is_503(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    class Down:
        def invoke(self, msgs): raise RuntimeError("429")
    monkeypatch.setattr(graph, "llm", Down())
    r = TestClient(main.app).post("/api/chat", json={"message": "hi"})
    assert r.status_code == 503 and "busy" in r.json()["detail"]


def test_vector_search_matches_meaning_and_arabic():
    assert search.search_projects("seaside home with private beach")  # no shared keywords needed
    assert all("Trump" in p["title"] for p in search.search_projects("مشاريع ترامب في عمان"))  # Arabic -> English pages
    assert search.search_projects("what is the price of bread") == []  # unrelated -> no project text sent


def test_listing_ranking_uses_meaning():
    hits, _ = search.search_listings({"city": "Riyadh", "purpose": "sale"}, "villa", k=5)
    assert sum("villa" in (l["sub_type"] or "").lower() for l in hits) >= 3


def test_repeat_question_is_answered_from_cache(monkeypatch):
    calls = []

    class Counting(FakeLLM):
        def stream(self, msgs):
            calls.append(1)
            yield from super().stream(msgs)
    monkeypatch.setattr(graph, "llm", Counting())
    first = graph.chat("2 bed for rent in Riyadh", "c1")
    again = graph.chat("2 bed for rent in Riyadh", "c2")  # new chat, same prompt
    assert first == again and len(calls) == 1


def test_link_guard_removes_invented_urls():
    allowed = {"https://wasalt.sa/en/property/1"}
    out = graph.guard_links("See https://wasalt.sa/en/property/1. Or https://fake.example/x, ok", allowed)
    assert "https://wasalt.sa/en/property/1." in out and "fake.example" not in out and "(link removed)," in out


def test_link_guard_works_on_streamed_pieces():
    sent = []
    g = graph._LineGuard(sent.append, {"https://ok.sa/a"})
    for piece in ["- https://ok", ".sa/a\n- https://inv", "ented.sa/b\nend"]:  # URLs split across pieces
        g(piece)
    g.flush()
    text = "".join(sent)
    assert "https://ok.sa/a" in text and "invented" not in text and text.endswith("end")


def test_lead_score_and_missing_fields():
    f = graph._parse_json('{"intent":"lead","project":"Trump Cliff Villas","max_price":3000000,"currency":"aed","timeline_months":3,"buyer_purpose":"Invest"}')
    lead = graph.score_lead(f)
    assert lead["label"] == "hot" and lead["score"] == 100 and lead["missing"] == [] and lead["budget"] == "3,000,000 AED"
    cold = graph.score_lead(graph._parse_json('{"intent":"lead"}'))
    assert cold["label"] == "cold" and cold["missing"][0] == "budget"


def test_lead_turn_adds_qualification_to_context(monkeypatch):
    class Lead(FakeLLM):
        filters = '{"source":"darglobal","intent":"lead","project":"Trump Cliff Villas"}'
    monkeypatch.setattr(graph, "llm", Lead())
    reply = graph.chat("I want to buy a Trump villa", "lead1")
    assert "LEAD QUALIFICATION" in reply and "Missing: budget" in reply  # FakeLLM echoes its context
    assert graph.graph.get_state({"configurable": {"thread_id": "lead1"}}).values["lead"]["location"] == "Trump Cliff Villas"


def test_currency_budget_is_converted_to_sar(monkeypatch):
    assert currency.to_sar(1000, "USD") == 3750 and currency.to_sar(3672.5, "AED") == 3750  # pegged, exact

    class Usd(FakeLLM):
        filters = '{"source":"wasalt","city":"Riyadh","purpose":"sale","max_price":200000,"currency":"USD"}'
    monkeypatch.setattr(graph, "llm", Usd())
    reply = graph.chat("homes in Riyadh under $200k", "usd1")
    prices = [int(x.replace(",", "")) for x in re.findall(r"\| ([\d,]+) SAR", reply)]
    assert prices and max(prices) <= 750_000 and "USD" in reply  # 200k USD = 750k SAR, shown with USD conversion


def test_lowest_reasoning_per_model():
    assert graph.lowest_reasoning({"mandatory": True, "supported_efforts": ["high"]}) == {"effort": "high"}  # Ernie: fixed
    assert graph.lowest_reasoning({"mandatory": False, "supported_efforts": ["high", "medium", "none"]}) == {"effort": "none"}
    assert graph.lowest_reasoning({"mandatory": True, "supported_efforts": None}) == {"effort": "minimal"}  # never "none"
    assert graph.lowest_reasoning(None) is None  # no reasoning controls: send nothing


def test_small_talk_skips_the_model_and_keeps_filters(monkeypatch):
    seen = []

    class Recording(FakeLLM):
        def invoke(self, msgs):
            seen.append(msgs)
            return super().invoke(msgs)
    monkeypatch.setattr(graph, "llm", Recording())
    graph.chat("3 bed for sale in Jeddah please", "st1")
    n = len(seen)
    graph.chat("thanks!", "st1")
    extract_calls = [m for m in seen[n:] if m[0].content.startswith("Turn the user's")]
    assert extract_calls == []  # greeting answered without the filter call
    assert graph.graph.get_state({"configurable": {"thread_id": "st1"}}).values["filters"]["city"] == "Jeddah"


def test_extract_sees_previous_filters_and_last_reply(monkeypatch):
    seen = []

    class Recording(FakeLLM):
        def invoke(self, msgs):
            seen.append(msgs)
            return super().invoke(msgs)
    monkeypatch.setattr(graph, "llm", Recording())
    graph.chat("3 bed for sale in Jeddah", "ex1")
    first_reply = graph.graph.get_state({"configurable": {"thread_id": "ex1"}}).values["messages"][-1].content
    graph.chat("tell me more about the third one", "ex1")
    last_extract = [m for m in seen if m[0].content.startswith("Turn the user's")][-1]
    assert '"city": "Jeddah"' in last_extract[0].content  # previous filters in the system message
    assert last_extract[-2].type == "ai" and last_extract[-2].content == first_reply  # latest reply whole: "third one" resolves


def test_unknown_currency_skips_price_filter_instead_of_reading_sar(monkeypatch):
    class Egp(FakeLLM):
        filters = '{"source":"wasalt","city":"Riyadh","purpose":"sale","max_price":100000,"currency":"egp"}'
    monkeypatch.setattr(graph, "llm", Egp())
    reply = graph.chat("homes in Riyadh under 100,000 EGP", "egp1")
    assert "No exchange rate for EGP" in reply
    assert any(int(x.replace(",", "")) > 100_000 for x in re.findall(r"\| ([\d,]+) SAR", reply))  # not cut at 100k SAR


def test_link_guard_keeps_links_before_arabic_punctuation():
    url = "https://wasalt.sa/en/property/sale/villa-5893874"
    out = graph.guard_links(f"فيلا: {url}، السعر 1,100,000؛ {url}؛", {url})
    assert "(link removed)" not in out and out.count(url) == 2


def test_bad_thread_id_and_blank_message_are_rejected():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    assert client.post("/api/chat", json={"message": "hi", "thread_id": "a\r\nSet-Cookie: x"}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "thread_id": "x:*"}).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422


def test_timeline_now_is_not_missing():
    lead = graph.score_lead(graph._parse_json('{"intent":"lead","timeline_months":0}'))
    assert "timeline_months" not in lead["missing"]


def test_price_format_never_shows_exponent():
    assert [currency.fmt(x) for x in (999, 999.6, 76_500, 999_499, 999_600, 1_234_567, 999_600_000)] == \
        ["999", "1K", "76.5K", "999K", "1M", "1.23M", "1B"]


def test_history_keeps_latest_reply_whole():
    from langchain_core.messages import HumanMessage
    long = "x" * 1000
    msgs = [HumanMessage("a"), AIMessage(long), HumanMessage("b"), AIMessage(long), HumanMessage("c")]
    out = graph._history(msgs)
    assert len(out[1].content) == 301 and out[3].content == long


def test_company_questions_get_company_pages(monkeypatch):
    assert search.search_company("where is your London office")[0]["title"] == "Contact and office addresses"
    assert "payment plans" in search.search_company("ما هي خطط الدفع المتاحة؟")[0]["title"]  # Arabic -> English FAQ

    class Company(FakeLLM):
        filters = '{"source":"darglobal","keywords":["golden visa"]}'
    monkeypatch.setattr(graph, "llm", Company())
    assert "Golden Visa" in graph.chat("can I get a golden visa if I buy in Dubai?", "co1")  # FakeLLM echoes its context


def test_api_keys_are_tried_in_order():
    env = {"OPENROUTER_API_KEY_10": "k10", "OPENROUTER_API_KEY_2": " k2 ", "OPENROUTER_API_KEY_1": "k1",
           "OPENROUTER_API_KEY": "", "OPENROUTER_API_KEY_3": "k1", "OTHER": "x"}
    assert graph.api_keys(env) == ["k1", "k2", "k10"]  # numeric order, trimmed, blank and repeated keys skipped
    assert graph.api_keys({}) == []


def test_next_key_only_after_an_error_from_openrouter():
    import httpx
    import openai
    from langchain_core.runnables import RunnableLambda
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    tried = []

    def make(model, key):
        def call(_):
            tried.append((key, model))
            if key == "k1":
                raise errors[model]
            return f"{key}:{model}"
        return RunnableLambda(call)

    errors = {m: openai.RateLimitError("limit", response=httpx.Response(429, request=req), body=None) for m in "ab"}
    assert graph.build_llm(["k1", "k2"], ["a", "b"], make).invoke("hi") == "k2:a"  # daily limit on k1 -> k2
    errors, tried[:] = {m: openai.APITimeoutError(request=req) for m in "ab"}, []  # provider slow for everyone
    try:
        graph.build_llm(["k1", "k2"], ["a", "b"], make).invoke("hi")
        raise AssertionError("expected the timeout")
    except openai.APITimeoutError:
        assert tried == [("k1", "a"), ("k1", "b")]  # a timeout is as slow on k2: don't wait twice
