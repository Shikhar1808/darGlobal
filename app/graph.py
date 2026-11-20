"""LangGraph flow: extract (LLM -> JSON filters) -> retrieve (hybrid search) -> [qualify (leads)] -> answer (LLM, streamed)."""
import json
import logging
import os
import re
import urllib.request
from typing import Annotated, TypedDict

import openai
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import Generation
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import currency, search

log = logging.getLogger("uvicorn.error")
MODEL = os.getenv("MODEL", "baidu/ernie-5.5-flash-preview:free")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
HISTORY = 6  # messages of chat history sent to both model calls


def _model_info():
    """Reasoning capabilities per model, from OpenRouter's public model list (no key, no quota)."""
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=5) as r:
            return {m["id"]: m.get("reasoning") for m in json.load(r)["data"]}
    except Exception as e:
        log.warning("OpenRouter model list unavailable (%s); using each model's default reasoning", e)
        return {}


EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]  # lowest first


def lowest_reasoning(info):
    """Reasoning tokens are slow and count as output, but our two steps don't need them: ask for the lowest allowed."""
    if not info:
        return None  # no reasoning controls: send nothing
    allowed = info.get("supported_efforts") or EFFORTS
    if info.get("mandatory"):
        allowed = [e for e in allowed if e != "none"]
    lowest = min(allowed, key=EFFORTS.index, default=None)
    return {"effort": lowest} if lowest else None


MODEL_INFO = _model_info()


def api_keys(env=os.environ):
    """OPENROUTER_API_KEY, then OPENROUTER_API_KEY_1, _2, ... in that order, blanks and repeats skipped."""
    names = ["OPENROUTER_API_KEY"] + sorted((n for n in env if re.fullmatch(r"OPENROUTER_API_KEY_\d+", n)),
                                            key=lambda n: int(n.rsplit("_", 1)[1]))
    return list(dict.fromkeys(env[n].strip() for n in names if env.get(n, "").strip()))


KEYS = api_keys()


def _llm(model, key):
    reasoning = lowest_reasoning(MODEL_INFO.get(model))
    return ChatOpenAI(model=model, base_url="https://openrouter.ai/api/v1", api_key=key,
                      temperature=0.2, timeout=25, max_retries=0,  # the fallbacks are the retry
                      extra_body={"reasoning": reasoning} if reasoning else None)


def build_llm(keys, models, make=_llm):
    """Both models on the first key, then on the next key. Free models get rate-limited and the daily cap is per key.
    Keys switch only on an error OpenRouter sends back (limit, bad key, no credit): a timeout is as slow on any key."""
    per_key = []
    for key in keys:
        first, *rest = [make(m, key) for m in dict.fromkeys(models)]
        per_key.append(first.with_fallbacks(rest) if rest else first)
    first, *rest = per_key
    return first.with_fallbacks(rest, exceptions_to_handle=(openai.APIStatusError,)) if rest else first


llm = build_llm(KEYS or ["not-set"], [MODEL, FALLBACK_MODEL])

# Cache of model calls (same prompt -> same reply, no model call) and per-thread chat memory.
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_URL:
    from langchain_redis import RedisCache
    from langgraph.checkpoint.redis import ShallowRedisSaver

    cache = RedisCache(redis_url=REDIS_URL, ttl=7 * 24 * 3600, prefix="darglobal:llm")  # data is a fixed snapshot
    memory = ShallowRedisSaver(REDIS_URL, ttl={"default_ttl": 24 * 60, "refresh_on_read": True})  # chats expire after 24 h idle
    memory.setup()  # creates the Redis search indexes once
else:
    # shortcut: in-process cache and memory: lost on restart, memory never evicted. Set REDIS_URL for anything real.
    cache, memory = InMemoryCache(maxsize=1000), MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    filters: dict
    context: str
    lead: dict


EXTRACT_PROMPT = """Turn the user's latest message into search filters for a property chatbot. Use the chat history for follow-ups ("cheaper ones?") and keep facts the user gave earlier (budget, timeline, purpose).
Data: DarGlobal = luxury branded developments (Dubai, RAK, Oman/AIDA, Qatar, Spain, London, Riyadh, Jeddah) and the company itself (about, FAQ, buying process, payment plans, residency visas, offices, brand partners, hotels). Wasalt = individual Saudi listings for sale/rent.
Return ONLY a JSON object with these keys (null when unknown):
{"source": "darglobal" | "wasalt" | "both" | "none",   // none = greeting/small talk only
 "intent": "search" | "lead",                           // lead = wants to buy, book a visit, or be contacted
 "city": string, "purpose": "sale" | "rent", "property_type": "Apartment" | "Villa" | "Land" | "Floor" | "Building" | "Rest House" | "Room",
 "bedrooms": int, "min_bedrooms": int, "min_price": int, "max_price": int,   // the user's budget as plain integers (500k -> 500000, 1.2M -> 1200000)
                                                        // rent prices are PER YEAR: convert a monthly budget to yearly (x12)
 "currency": "SAR" | "USD" | "AED" | "GBP" | "EUR" | "QAR" | "OMR" | "INR",  // currency of the budget or the one the user asks for; null = SAR
 "project": string,                                     // DarGlobal project name if the user names one
 "timeline_months": int,                                // when they want to buy/move (e.g. "this year" -> 12, "asap" -> 1)
 "buyer_purpose": "live" | "invest",                   // own use or investment, only if the user says so
 "keywords": [string]}                                  // other important words: district, brand, amenity
Use "both" when unsure whether the user means DarGlobal projects or Saudi listings."""

ANSWER_PROMPT = """You are the property assistant for DarGlobal (luxury branded developments) and Wasalt (Saudi listings).
Rules:
- Answer ONLY from the CONTEXT below. Never invent projects, prices, sizes or links. If the context has no match, say so and suggest how to broaden the search.
- Give concrete facts (price, bedrooms, area, location) and the link for every property you mention. If a price has an "about ... <currency>" conversion, include it.
- When you use Wasalt listings, end with "Source: Wasalt.sa". Prices and availability change; point users to the live link.
- If the user wants to buy, visit or be contacted: follow the LEAD QUALIFICATION section, and send them to the project's page (it has a Register Interest form) or https://darglobal.co.uk/get-in-touch for DarGlobal, or to the listing link for Wasalt. Never ask for their name, phone or email.
- Reply in the same language as the user's last message (Arabic or English). Be concise; use short "- " lists for multiple properties.
- Plain text only: no markdown (no **bold**, no [text](url)); write full URLs.

CONTEXT:
{context}"""

URL = re.compile(r"https?://[^\s<>\")\]،؛]+")  # stops at Arabic comma/semicolon too
HOMEPAGES = {"https://darglobal.co.uk", "https://wasalt.sa", "https://wasalt.sa/en"}


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        f = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        f = {}
    for k in ("bedrooms", "min_bedrooms", "min_price", "max_price", "timeline_months"):
        try:
            f[k] = int(float(f[k])) if f.get(k) not in (None, "") else None
        except (TypeError, ValueError):
            f[k] = None
    for k in ("city", "purpose", "property_type", "project", "buyer_purpose", "currency", "intent"):  # models return lists, odd casing
        v = f.get(k)
        f[k] = v.strip() if isinstance(v, str) and v.strip() else None
    f["purpose"] = f["purpose"].lower() if f["purpose"] else None
    f["intent"] = "lead" if (f["intent"] or "").lower() == "lead" else "search"
    f["buyer_purpose"] = f["buyer_purpose"].lower() if (f["buyer_purpose"] or "").lower() in ("live", "invest") else None
    # keep any 3-letter code: retrieve skips the price filter if it can't convert it, rather than read it as SAR
    f["currency"] = f["currency"].upper() if re.fullmatch(r"[A-Za-z]{3}", f["currency"] or "") else None
    f["source"] = f.get("source") if f.get("source") in ("darglobal", "wasalt", "both", "none") else "both"
    f["keywords"] = [str(k) for k in f.get("keywords") or []][:8]
    return f


def _ask(msgs, write=None):
    """Model call through the cache. With `write`, the reply is streamed piece by piece to chat_stream.
    LangChain's built-in cache can't be used: it keys on message ids, which LangGraph makes unique per message."""
    key = json.dumps([(m.type, m.content) for m in msgs], ensure_ascii=False)  # whole prompt incl. history
    if hit := cache.lookup(key, MODEL):
        text = hit[0].text
        if write:
            write(text)
        return text
    if write:
        text = ""
        for chunk in llm.stream(msgs):
            write(chunk.content)
            text += chunk.content
    else:
        text = llm.invoke(msgs).content
    if text:
        cache.update(key, MODEL, [Generation(text=text)])
    return text


def guard_links(text, allowed):
    """Replaces any URL the search did not return, so the bot cannot send users to invented pages."""
    def check(m):
        url = m.group(0).rstrip(".,;:!?'")
        if url.rstrip("/") in allowed:
            return m.group(0)
        log.warning("removed invented link: %s", url)
        return "(link removed)" + m.group(0)[len(url):]
    return URL.sub(check, text)


class _LineGuard:
    """Streams text line by line, checking each finished line's links before it is sent."""

    def __init__(self, write, allowed):
        self.write, self.allowed, self.buf = write, allowed, ""

    def __call__(self, piece):
        self.buf += piece
        if "\n" in self.buf:
            done, self.buf = self.buf.rsplit("\n", 1)
            self.write(guard_links(done + "\n", self.allowed))

    def flush(self):
        self.write(guard_links(self.buf, self.allowed))


SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks?|thank you|ok(ay)?|bye|good (morning|afternoon|evening)|"
                        r"مرحبا|أهلا|اهلا|السلام عليكم|شكرا|شكراً)[\s!.,?؟]*$", re.I)


def extract(state: State):
    """Previous turn's filters + trimmed chat history (the latest reply whole, so "the third one" resolves)."""
    prev = {k: v for k, v in (state.get("filters") or {}).items() if v not in (None, [], "") and k not in ("source", "intent")}
    if SMALL_TALK.match(state["messages"][-1].content):  # greeting or thanks: no model call needed
        return {"filters": {**_parse_json("{}"), **prev, "source": "none", "intent": "search"}}
    system = EXTRACT_PROMPT + (f"\n\nFilters from the previous turn: {json.dumps(prev, ensure_ascii=False)}" if prev else "")
    return {"filters": _parse_json(_ask([SystemMessage(system), *_history(state["messages"])]))}  # bad JSON -> broad search


def _history(messages):
    """Recent messages for both model calls. Older replies are cut to 300 characters;
    the latest is kept whole so "tell me about the second one" still works."""
    recent = messages[-HISTORY:]
    last_ai = max((i for i, m in enumerate(recent) if m.type == "ai"), default=None)
    return [AIMessage(m.content[:300] + "…") if m.type == "ai" and i != last_ai and len(m.content) > 300 else m
            for i, m in enumerate(recent)]


def retrieve(state: State):
    f, question = dict(state["filters"]), state["messages"][-1].content
    cur = f["currency"] if f["currency"] and f["currency"] != "SAR" else None
    price_note = ""
    if cur and cur not in currency.PER_USD:  # no rate (unknown code, or live rates failed to load)
        if f["min_price"] or f["max_price"]:
            price_note = f"\nNo exchange rate for {cur}, so the price filter was skipped and prices are in SAR. Tell the user."
        f["min_price"] = f["max_price"] = cur = None
    for k in ("min_price", "max_price"):  # the search works in SAR
        if f[k] and cur:
            f[k] = currency.to_sar(f[k], cur)
    parts, cards = [], []
    if f["source"] in ("darglobal", "both"):
        projects = search.search_projects(f"{question} {f['project'] or ''}", f["keywords"])
        parts.append("## DarGlobal projects (all)\n" + search.PROJECT_INDEX)
        if projects:
            parts.append("## Best-matching DarGlobal project details\n" + "\n\n".join(map(search.project_block, projects)))
        cards += [search.project_card(p) for p in projects]
        # only for DarGlobal questions: scores can't tell "villas in Riyadh" apart from company questions
        company = search.search_company(question, f["keywords"], k=2) if f["source"] == "darglobal" else []
        if company:
            parts.append("## About DarGlobal (company pages)\n" + "\n\n".join(map(search.company_block, company)))
    if f["source"] in ("wasalt", "both"):
        listings, relaxed = search.search_listings(f, question)
        head = "## Wasalt listings (Source: Wasalt.sa)" + price_note
        if relaxed:
            head += f"\nNo exact match; these ignore the {', '.join(relaxed)} filter(s). Tell the user."
        parts.append(head + "\n" + ("\n".join(search.listing_line(l, cur) for l in listings) or "No listings match."))
        cards += [search.listing_card(l, cur) for l in listings]
    if cards:
        get_stream_writer()({"cards": cards})  # the page shows the ones the answer mentions
    return {"context": "\n\n".join(parts)}


def score_lead(f):
    """Rule-based lead score, no model call: what a sales team needs before calling back."""
    lead = {"budget": f["max_price"] and f"{f['max_price']:,} {f['currency'] or 'SAR'}",
            "location": f["project"] or f["city"], "timeline_months": f["timeline_months"], "purpose": f["buyer_purpose"]}
    t = f["timeline_months"]
    score = (10 + 25 * bool(lead["budget"]) + 25 * bool(lead["location"]) + 10 * bool(lead["purpose"])
             + (30 if t is not None and t <= 6 else 20 if t is not None and t <= 12 else 10 if t else 0))
    lead |= {"score": score, "label": "hot" if score >= 70 else "warm" if score >= 40 else "cold",
             "missing": [k for k in ("budget", "location", "timeline_months", "purpose") if lead[k] is None]}  # 0 months = now
    return lead


def qualify(state: State):
    lead = score_lead(state["filters"])
    log.info("lead %s", json.dumps(lead))  # shortcut: logged only; a real setup posts this (no contact data) to the CRM
    known = ", ".join(f"{k}={v}" for k, v in lead.items() if k in ("budget", "location", "timeline_months", "purpose") and v)
    if lead["missing"]:
        todo = (f"Missing: {', '.join(lead['missing'])}. Answer their question, then ask ONE short question about "
                f"'{lead['missing'][0]}' so the sales team can help faster.")
    else:
        todo = "All known. Give a 2-3 line summary they can paste into the Register Interest form, plus the link."
    note = f"## LEAD QUALIFICATION (user wants to buy, visit or be contacted)\nKnown: {known or 'nothing yet'}.\n{todo}"
    return {"lead": lead, "context": f"{state['context']}\n\n{note}"}


def answer(state: State):
    context = state.get("context") or "(no search needed)"
    msgs = [SystemMessage(ANSWER_PROMPT.format(context=context)), *_history(state["messages"])]
    allowed = HOMEPAGES | {u.rstrip(".,;:!?'/") for u in URL.findall(msgs[0].content)}
    guard = _LineGuard(get_stream_writer(), allowed)
    text = _ask(msgs, write=guard)
    guard.flush()
    return {"messages": [AIMessage(guard_links(text, allowed))]}


def after_extract(state: State):
    f = state["filters"]
    return "answer" if f["source"] == "none" and f["intent"] != "lead" else "retrieve"


def after_retrieve(state: State):
    return "qualify" if state["filters"]["intent"] == "lead" else "answer"


builder = StateGraph(State)
builder.add_node("extract", extract)
builder.add_node("retrieve", retrieve)
builder.add_node("qualify", qualify)
builder.add_node("answer", answer)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", after_extract, ["retrieve", "answer"])
builder.add_conditional_edges("retrieve", after_retrieve, ["qualify", "answer"])
builder.add_edge("qualify", "answer")
builder.add_edge("answer", END)
graph = builder.compile(checkpointer=memory)


def chat_stream(message: str, thread_id: str):
    """Yields reply text pieces (str) and, before them, result cards ({"cards": [...]})."""
    yield from graph.stream({"messages": [HumanMessage(message)], "context": "", "lead": {}},
                            {"configurable": {"thread_id": thread_id}}, stream_mode="custom")


def chat(message: str, thread_id: str) -> str:
    return "".join(p for p in chat_stream(message, thread_id) if isinstance(p, str))
