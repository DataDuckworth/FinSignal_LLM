"""Tests for the LLM step. Nothing here calls the real API: a fake client replays scripted replies."""
import json
import types

import groq
import httpx
import pytest

from src import llm_extract as llm

GOOD = '{"sentiment":"positive","score":0.4,"event_type":"earnings","confidence":0.8}'
REQUEST = httpx.Request("POST", "https://api.groq.com/test")


def status_error(cls, status, message, headers=None):
    return cls(message, response=httpx.Response(status, request=REQUEST, headers=headers or {}), body=None)


class FakeClient:
    """Mimics client.chat.completions.create(...) and replays scripted replies (text, or an exception to raise)."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=reply))])


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Fresh cache folder per test, no pacing and no real sleeping."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "REQUESTS_PER_MINUTE", 10 ** 9)
    monkeypatch.setattr(llm, "TOKENS_PER_MINUTE", 10 ** 9)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("reply, accepted", [
    (GOOD, True),
    ("```json\n" + GOOD + "\n```", True),
    ("Sure! " + GOOD + " Hope that helps.", True),
    ('{"sentiment":"Positive","score":0.1,"event_type":"M&A","confidence":0.5}', True),
    ('{"sentiment":"bullish","score":0.4,"event_type":"earnings","confidence":0.8}', False),
    ('{"sentiment":"positive","score":1.7,"event_type":"earnings","confidence":0.8}', False),
    ('{"sentiment":"positive","score":0.4,"event_type":"earnings"}', False),
    ('{"sentiment":"positive","score":NaN,"event_type":"earnings","confidence":0.8}', False),
    ("The headline looks positive.", False),
    ("", False),
])
def test_parse_response(reply, accepted):
    if accepted:
        assert llm.parse_response(reply)["sentiment"] in llm.SENTIMENTS
    else:
        with pytest.raises(ValueError):
            llm.parse_response(reply)


def batch_reply(*entries):
    return json.dumps({"results": [{"id": i, "sentiment": s, "score": sc, "event_type": "earnings", "confidence": 0.8}
                                   for i, s, sc in entries]})


def test_batch_answers_are_matched_by_id_not_by_order():
    out = llm.parse_batch_response(batch_reply((3, "neutral", 0.3), (1, "positive", 0.1), (2, "negative", -0.2)), 3)
    assert [out[p]["score"] for p in range(3)] == [0.1, -0.2, 0.3]


def test_batch_duplicate_and_unknown_ids():
    out = llm.parse_batch_response(batch_reply((1, "positive", 0.1), (1, "negative", -0.1), (9, "neutral", 0.0)), 2)
    assert out[0]["ok"] is False and 1 not in out                  # duplicate id: rejected; unknown id 9: ignored


# ---------------------------------------------------------------- single mode
def test_valid_answer_is_cached_and_second_call_needs_no_request():
    client = FakeClient([GOOD])
    assert llm.extract("h1", "T", client=client)["ok"]
    again = FakeClient([])
    assert llm.extract("h1", "T", client=again)["ok"] and again.calls == 0


def test_bad_answers_are_not_cached():
    client = FakeClient(["no json", "still no json"])
    result = llm.extract("h2", "T", client=client)
    assert not result["ok"] and client.calls == 2
    assert not llm.cache_path("h2", "T", False).exists()


def test_rate_limit_is_retried_after_the_requested_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    err = status_error(groq.RateLimitError, 429, "rate limited", {"retry-after": "3"})
    assert llm.extract("h3", "T", client=FakeClient([err, GOOD]))["ok"]
    assert 3.0 in waits


def test_long_rate_limit_wait_stops_the_run():
    err = status_error(groq.RateLimitError, 429, "tokens per day (TPD) limit", {"retry-after": "7000"})
    with pytest.raises(llm.RateLimitExhausted):
        llm.extract("h4", "T", client=FakeClient([err]))


def test_authentication_error_is_raised_not_swallowed():
    with pytest.raises(groq.AuthenticationError):
        llm.extract("h5", "T", client=FakeClient([status_error(groq.AuthenticationError, 401, "bad key")]))


def test_groq_json_mode_failure_counts_as_a_bad_answer_not_a_crash():
    err = status_error(groq.BadRequestError, 400, "Failed to generate JSON (json_validate_failed)")
    client = FakeClient([err, err])
    result = llm.extract("h6", "T", client=client)
    assert not result["ok"] and client.calls == 2
    # one failure followed by a good answer is accepted
    assert llm.extract("h7", "T", client=FakeClient([err, GOOD]))["ok"]


def test_other_bad_requests_are_still_raised():
    err = status_error(groq.BadRequestError, 400, "The model `x` has been decommissioned")
    with pytest.raises(groq.BadRequestError):
        llm.extract("h8", "T", client=FakeClient([err]))


def test_corrupt_cache_file_is_ignored_and_replaced():
    llm.cache_path("h9", "T", False).write_text('{"sentiment": "pos', encoding="utf-8")     # half-written file
    client = FakeClient([GOOD])
    assert llm.extract("h9", "T", client=client)["ok"] and client.calls == 1
    assert json.loads(llm.cache_path("h9", "T", False).read_text(encoding="utf-8"))["ok"] is True
    assert not list(llm.CACHE_DIR.glob("*.tmp"))                    # no temporary files left behind


def test_cache_key_depends_on_few_shot_ticker_and_mode():
    assert llm.cache_path("h", "T", True) != llm.cache_path("h", "T", False)
    assert llm.cache_path("h", "A", False) != llm.cache_path("h", "B", False)
    assert llm.cache_path("h", "T", False, "batch") != llm.cache_path("h", "T", False)


def test_user_message_contains_no_date():
    assert "date" not in llm.user_message("Acme raises guidance", "ACME").lower()


# ---------------------------------------------------------------- batch mode
def test_batch_happy_path_then_fully_cached():
    items = [("b1", "T"), ("b2", "T"), ("b3", "T")]
    client = FakeClient([batch_reply((1, "positive", 0.1), (2, "negative", -0.2), (3, "neutral", 0.0))])
    assert all(r["ok"] for r in llm.extract_batch(items, client=client, batch_size=3)) and client.calls == 1
    again = FakeClient([])
    assert all(r["ok"] for r in llm.extract_batch(items, client=again, batch_size=3)) and again.calls == 0


def test_batch_missing_item_is_resent_and_invalid_item_stays_failed_if_still_bad():
    client = FakeClient([batch_reply((1, "positive", 0.1), (3, "neutral", 0.0)), batch_reply((1, "negative", -0.5))])
    out = llm.extract_batch([("m1", "T"), ("m2", "T"), ("m3", "T")], client=client, batch_size=3)
    assert client.calls == 2 and all(r["ok"] for r in out) and out[1]["score"] == -0.5

    client = FakeClient([batch_reply((1, "positive", 0.1), (2, "negative", 1.7)), batch_reply((1, "negative", 1.9))])
    out = llm.extract_batch([("f1", "T"), ("f2", "T")], client=client, batch_size=2)
    assert out[0]["ok"] and not out[1]["ok"]
    assert not llm.cache_path("f2", "T", False, "batch").exists()   # failures are never cached


def test_batch_json_mode_failure_does_not_crash_the_run():
    err = status_error(groq.BadRequestError, 400, "json_validate_failed")
    client = FakeClient([err, err, err, err])
    out = llm.extract_batch([("j1", "T"), ("j2", "T")], client=client, batch_size=2)
    assert not any(r["ok"] for r in out)
