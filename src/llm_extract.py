"""Headline -> structured JSON with an LLM served by Groq: validated, cached on disk, rate-limit aware.

Two ways to call the model
- extract():        one headline per request (simple; 1 request per headline).
- extract_batch():  several headlines per request, answers matched back by item id. Needed because the free tier
                    caps requests per day (measured: 1,000/day, 8,000 tokens/min and 200,000 tokens/day for
                    openai/gpt-oss-20b on 2026-11-21). The token cap is the one that binds in practice: about
                    1,400 headlines per day in batches of 10.

Design choices (and why)
- temperature = 0: the same input should give the same answer, otherwise a re-run changes the results.
- Every answer is parsed AND validated (allowed labels, score range). An invalid answer is never turned into a
  fake "neutral" row. It is returned as {"ok": False, "error": ...} so the caller can count it.
- Only valid answers are written to the cache, so a failure is retried on the next run instead of being
  remembered forever. In batch mode the cache is per item, so a half-finished run resumes where it stopped.
- The cache key contains model, prompt version, mode, few-shot flag, ticker and headline.
- No date is sent to the model, only the ticker and the headline.
- If the daily request limit is hit, RateLimitExhausted is raised instead of sleeping for hours; everything done so
  far is already cached, so the run can simply be repeated later.

Batching caveat: the model sees several items at once, so an answer could be influenced by its neighbours. The prompt
says to judge each item independently, and notebook 03 compares batched with single-item answers on the same
sentences. Batch mode is zero-shot only.
"""
import hashlib
import json
import math
import os
import pathlib
import re
import time

import groq

from . import config

config.load_env()      # fills os.environ from the git-ignored .env file (never overrides what is already set)

# Anchored to this file, not to the current folder: a notebook running in notebooks/ must not create its own cache.
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "llm_cache"

# Groq retires models over time (llama-3.1-8b-instant and llama-3.3-70b-versatile were announced for removal
# in Aug 2026), so the model is configurable and the notebook checks it against the live model list.
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
PROMPT_VERSION = "v1"                      # bump when SYSTEM or EXAMPLES change
BATCH_PROMPT_VERSION = "b1"                # bump when BATCH_SYSTEM changes
MAX_TOKENS = 600                           # single mode: generous, reasoning models think before the JSON

# Limits measured from the x-ratelimit-* response headers (openai/gpt-oss-20b, free tier, 2026-11-21). They are
# model- and account-specific and may change: re-check the headers if you switch model.
REQUESTS_PER_MINUTE = 20                   # conservative pacing between live calls
TOKENS_PER_MINUTE = 8000                   # measured limit; pacing keeps usage at 90% of it
MAX_WAIT_SECONDS = 180                     # a longer rate-limit wait (e.g. the daily cap) stops the run instead

SENTIMENTS = ("positive", "neutral", "negative")
EVENT_TYPES = ("earnings", "guidance", "m&a", "analyst_rating", "legal", "product", "macro", "other")

_HEAD = "You are a financial news analyst. You will be given one news headline and, if known, the stock ticker it is about.\n"
_SCHEMA = """Judge ONLY from the headline text. Return ONLY a JSON object with exactly these keys:
{"sentiment": "positive|neutral|negative",
 "score": <number from -1 to 1>,
 "event_type": "earnings|guidance|m&a|analyst_rating|legal|product|macro|other",
 "confidence": <number from 0 to 1>}
"""
_DEFINITIONS = """Definitions:
- sentiment: tone of the news for shareholders of that company.
- score: expected direction and size of the stock's move relative to the overall market over the next trading day,
  -1 = very negative, 0 = no information or no expected effect, +1 = very positive.
- event_type: the kind of event the headline reports; use "other" if none fits.
- confidence: how clear and informative the headline is, 0 = vague, 1 = unambiguous."""
SYSTEM = _HEAD + _SCHEMA + _DEFINITIONS     # byte-for-byte the v1 prompt, so cached v1 answers stay valid

BATCH_SYSTEM = """You are a financial news analyst. You will be given several numbered news items, each with a headline and, if known, the stock ticker it is about.
Judge each item ONLY from its own text and independently of the other items. Return ONLY a JSON object of the form
{"results": [{"id": <item number>, "sentiment": "...", "score": <number>, "event_type": "...", "confidence": <number>}, ...]}
with exactly one entry per item, using the item numbers you were given. Allowed values:
sentiment is one of positive, neutral, negative; score is a number from -1 to 1;
event_type is one of earnings, guidance, m&a, analyst_rating, legal, product, macro, other; confidence is a number from 0 to 1.
""" + _DEFINITIONS

# Invented companies and events (nothing taken from our data) so the examples cannot leak real outcomes.
EXAMPLES = [
    ("Zorvex", "Zorvex raises full-year profit forecast after record quarterly revenue",
     {"sentiment": "positive", "score": 0.7, "event_type": "guidance", "confidence": 0.9}),
    ("Quillon", "Quillon shares fall as regulator opens investigation into accounting practices",
     {"sentiment": "negative", "score": -0.7, "event_type": "legal", "confidence": 0.85}),
    ("Marnet", "Marnet to hold annual shareholder meeting on Tuesday",
     {"sentiment": "neutral", "score": 0.0, "event_type": "other", "confidence": 0.8}),
    ("Dexa", "Analyst at Brightside upgrades Dexa to Buy, lifts price target to $48",
     {"sentiment": "positive", "score": 0.4, "event_type": "analyst_rating", "confidence": 0.7}),
]


class RateLimitExhausted(RuntimeError):
    """The rate limit needs a long wait (typically the daily request cap). Cached work is kept; re-run later."""


_client = None
_last_call = 0.0
_last_tokens = 0
usage = {"requests": 0, "tokens": 0}       # running totals of live calls, to compare with the planning table


def get_client():
    """Create the Groq client on first use, so importing this module never needs a key."""
    global _client
    if _client is None:
        _client = groq.Groq(api_key=config.get_secret("GROQ_API_KEY"))     # raises a clear error if missing
    return _client


# ---------------------------------------------------------------- prompts
def user_message(headline, ticker=None):
    return (f"Ticker: {ticker}\n" if ticker else "") + f"Headline: {headline}"


def build_messages(headline, ticker=None, few_shot=False):
    messages = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for ex_ticker, ex_headline, ex_answer in EXAMPLES:
            messages.append({"role": "user", "content": user_message(ex_headline, ex_ticker)})
            messages.append({"role": "assistant", "content": json.dumps(ex_answer)})
    messages.append({"role": "user", "content": user_message(headline, ticker)})
    return messages


def build_batch_messages(items):
    """items: list of (headline, ticker or None). Items are numbered from 1."""
    lines = [f"{i}. " + (f"Ticker: {t} | " if t else "") + f"Headline: {h}" for i, (h, t) in enumerate(items, 1)]
    user = f"Score these {len(items)} items. Return one entry per item.\n\n" + "\n".join(lines)
    return [{"role": "system", "content": BATCH_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- parsing and validation
def _json_object(text):
    """Find the JSON object in a model reply (tolerates a code fence or chatter around it)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)   # models sometimes wrap JSON in a code fence
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])                                        # JSONDecodeError is a ValueError
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def _validated_fields(obj):
    """The four required fields, checked and normalised. Raises ValueError if anything is wrong."""
    sentiment = str(obj.get("sentiment", "")).strip().lower()
    event_type = str(obj.get("event_type", "")).strip().lower()
    if sentiment not in SENTIMENTS:
        raise ValueError(f"bad sentiment: {obj.get('sentiment')!r}")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"bad event_type: {obj.get('event_type')!r}")
    try:
        score, confidence = float(obj["score"]), float(obj["confidence"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing or non-numeric score / confidence")
    if not (math.isfinite(score) and -1.0 <= score <= 1.0):
        raise ValueError(f"score out of range: {score}")
    if not (math.isfinite(confidence) and 0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence out of range: {confidence}")
    return {"sentiment": sentiment, "score": score, "event_type": event_type, "confidence": confidence}


def parse_response(text):
    """Single-item reply -> validated dict. Raises ValueError if anything is wrong."""
    return _validated_fields(_json_object(text))


def parse_batch_response(text, n_items):
    """Batch reply -> {position (0-based): result dict}. Items are matched by their id, never by order.

    A result is {"ok": True, ...fields} or {"ok": False, "error": ...}. Missing items are simply absent from the
    dict. Raises ValueError if the reply has no usable "results" list at all.
    """
    results = _json_object(text).get("results")
    if not isinstance(results, list):
        raise ValueError("no 'results' list")
    out, duplicated = {}, set()
    for entry in results:
        if not isinstance(entry, dict):
            continue
        try:
            pos = int(entry.get("id")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= pos < n_items:                   # an id we never sent: ignore it
            continue
        if pos in out:
            duplicated.add(pos)
            continue
        try:
            out[pos] = {**_validated_fields(entry), "ok": True}
        except ValueError as err:
            out[pos] = {"ok": False, "error": f"bad item: {err}"}
    for pos in duplicated:                           # two answers for one item: we cannot know which is right
        out[pos] = {"ok": False, "error": "duplicate id in reply"}
    return out


# ---------------------------------------------------------------- calling the model
def _retry_after_seconds(err):
    try:
        return float(err.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _wait_after_rate_limit(err, attempt):
    """Sleep as long as the server asks, but stop the whole run if that is a long wait (the daily cap)."""
    wait = _retry_after_seconds(err)
    # The length of the wait decides, not the wording: a "tokens per day" limit that refills in 7 minutes is worth
    # sleeping through. Only when the server gives no wait at all do we fall back on the "per day" wording.
    if (wait is not None and wait > MAX_WAIT_SECONDS) or (wait is None and "per day" in str(err).lower()):
        server_says = " ".join(str(getattr(err, "message", err)).split())[:300]    # which limit was hit, in Groq's words
        raise RateLimitExhausted(
            (f"rate limit needs a long wait (about {wait / 60:.0f} min per the server)" if wait else "daily limit reached")
            + f". Server message: {server_says}") from err
    time.sleep(wait or min(2 ** attempt * 2, 60))


def _pace():
    """Stay under REQUESTS_PER_MINUTE and 90% of TOKENS_PER_MINUTE by sleeping before a live call."""
    global _last_call
    gap = max(60.0 / REQUESTS_PER_MINUTE, _last_tokens / (0.9 * TOKENS_PER_MINUTE) * 60)
    wait = gap - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _call_model(client, messages, max_tokens=MAX_TOKENS):
    """One live request. Returns (reply text, tokens used)."""
    global _last_tokens
    kwargs = dict(model=MODEL, messages=messages, temperature=0,
                  response_format={"type": "json_object"}, max_completion_tokens=max_tokens)
    if MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"          # reasoning models think first; keep that short and cheap
    response = client.chat.completions.create(**kwargs)
    tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
    _last_tokens = tokens
    usage["requests"] += 1
    usage["tokens"] += tokens
    return response.choices[0].message.content or "", tokens


def cache_path(headline, ticker, few_shot, mode="single"):
    if mode == "single":                              # unchanged format, so earlier single-item answers stay valid
        key = f"{MODEL}|{PROMPT_VERSION}|{few_shot}|{ticker}|{headline}"
    else:
        key = f"{MODEL}|{BATCH_PROMPT_VERSION}|{mode}|{few_shot}|{ticker}|{headline}"
    return CACHE_DIR / f"{hashlib.md5(key.encode('utf-8')).hexdigest()}.json"    # md5 only names files, it is not for security


def _read_cache(path):
    """The cached answer, or None. A missing OR unreadable file counts as 'not cached yet'.

    Why not just json.loads: a run that is stopped (Ctrl-C, a closed laptop) in the middle of a write could leave a
    half-written file, and that single file would then crash every later run at the same headline.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("ok") is True else None


def _write_cache(path, result):
    """Write to a temporary file first and then rename it, so a stopped run never leaves a half-written answer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, path)                             # atomic on the same drive


def _is_json_failure(err):
    """True if a Groq 400 error means "JSON mode could not produce valid JSON" (code json_validate_failed).

    That is a bad answer for this one headline, like any other unusable reply, not a wrong request. Any other 400
    (retired model, malformed request) is still raised, because retrying cannot fix it.
    """
    text = f"{getattr(err, 'message', '')} {err}".lower()
    return "json_validate_failed" in text or "failed_generation" in text or "failed to generate json" in text


def extract(headline, ticker=None, few_shot=False, client=None, max_retries=5):
    """Return {"ok": True, sentiment, score, event_type, confidence} or {"ok": False, "error": ...}.

    Authentication / bad-request errors (wrong key, retired model) are raised, not swallowed: retrying cannot fix them.
    """
    path = cache_path(headline, ticker, few_shot)
    cached = _read_cache(path)
    if cached is not None:
        return cached

    client = client or get_client()
    messages = build_messages(headline, ticker, few_shot)
    last_error, bad_answers = "", 0
    for attempt in range(max_retries):
        try:
            _pace()
            text, _ = _call_model(client, messages)
            result = parse_response(text)
        except ValueError as err:                    # the model answered, but not in the format we asked for
            last_error, bad_answers = f"bad answer: {err}", bad_answers + 1
            if bad_answers >= 2:                     # temperature 0, so more retries would repeat the same answer
                break
            continue
        except groq.BadRequestError as err:
            if not _is_json_failure(err):
                raise                                # a real bad request (e.g. retired model): retrying cannot fix it
            last_error, bad_answers = "json mode failed", bad_answers + 1
            if bad_answers >= 2:
                break
            continue
        except groq.RateLimitError as err:
            last_error = "rate limited"
            _wait_after_rate_limit(err, attempt)
            continue
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as err:
            last_error = f"{type(err).__name__}"
            time.sleep(min(2 ** attempt * 2, 60))
            continue
        result["ok"] = True
        _write_cache(path, result)                                 # only valid answers are cached
        return result
    return {"ok": False, "error": last_error or "gave up"}


def _request_batch(client, chunk, max_retries):
    """One batch request with retries. Returns {position: result}; empty if the reply was unusable."""
    messages = build_batch_messages(chunk)
    max_tokens = 400 + 100 * len(chunk)             # room for ~60 tokens of JSON per item plus a little reasoning
    bad_answers = 0
    for attempt in range(max_retries):
        try:
            _pace()
            text, _ = _call_model(client, messages, max_tokens)
            return parse_batch_response(text, len(chunk))
        except ValueError:
            bad_answers += 1
            if bad_answers >= 2:
                return {}
        except groq.BadRequestError as err:
            if not _is_json_failure(err):
                raise                                # a real bad request (e.g. retired model): retrying cannot fix it
            bad_answers += 1
            if bad_answers >= 2:
                return {}
        except groq.RateLimitError as err:
            _wait_after_rate_limit(err, attempt)
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError):
            time.sleep(min(2 ** attempt * 2, 60))
    return {}


def extract_batch(items, client=None, batch_size=10, max_retries=5):
    """items: list of (headline, ticker or None). Returns one result dict per item, in the same order.

    Already-cached items are not sent again. The rest go out `batch_size` at a time; items that fail
    validation are re-sent once, grouped together, and any that still fail come back as {"ok": False}.
    Valid answers are cached per item immediately, so an interrupted run loses nothing.
    """
    results = [None] * len(items)
    pending = []
    for i, (headline, ticker) in enumerate(items):
        cached = _read_cache(cache_path(headline, ticker, False, "batch"))
        if cached is not None:
            results[i] = cached
        else:
            pending.append(i)
    if not pending:
        return results

    client = client or get_client()
    for _round in range(2):                                      # round 2 only re-sends what round 1 could not resolve
        unresolved = []
        for start in range(0, len(pending), batch_size):
            chunk_idx = pending[start:start + batch_size]
            got = _request_batch(client, [items[i] for i in chunk_idx], max_retries)
            for pos, i in enumerate(chunk_idx):
                result = got.get(pos) or {"ok": False, "error": "missing from the batch reply"}
                results[i] = result
                if result["ok"]:
                    _write_cache(cache_path(items[i][0], items[i][1], False, "batch"), result)
                else:
                    unresolved.append(i)
        pending = unresolved
        if not pending:
            break
    return results
