import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.graph import KEYS, chat_stream

log = logging.getLogger("uvicorn.error")
if KEYS:
    log.info("%d OpenRouter key(s); each is tried in turn when one fails", len(KEYS))
else:
    log.warning("No OPENROUTER_API_KEY set: every chat request will fail")
app = FastAPI(title="DarGlobal & Wasalt property chatbot")
INDEX = Path(__file__).parent / "static" / "index.html"


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,64}$")  # goes into a header and Redis keys


BUSY = "The AI model is busy right now. Please try again in a minute."


@app.post("/api/chat")
async def api_chat(body: ChatIn):
    """Streams NDJSON lines: {"cards": [...]} (search results), then {"t": "..."} reply text pieces.
    The thread id comes back in the X-Thread-Id header."""
    message = body.message.strip()
    if not message:  # only spaces: don't spend two model calls on it
        raise HTTPException(422, "Please type a question.")
    parts = chat_stream(message, body.thread_id)
    try:  # wait for the first piece, so most failures (rate limits, timeouts) still get a real 503
        first = await run_in_threadpool(next, parts, "")
    except Exception:
        log.exception("chat failed")  # usually OpenRouter rate limit / timeout on free models
        raise HTTPException(503, BUSY)

    def line(part):
        return json.dumps({"t": part} if isinstance(part, str) else part, ensure_ascii=False) + "\n"

    def rest():
        yield line(first)
        try:
            for part in parts:
                yield line(part)
        except Exception:  # the model failed mid-reply: the 200 status is already sent, so say it in the text
            log.exception("chat failed mid-stream")
            yield line("\n\n" + BUSY)

    return StreamingResponse(rest(), media_type="application/x-ndjson",
                             headers={"X-Thread-Id": body.thread_id, "X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(INDEX)
